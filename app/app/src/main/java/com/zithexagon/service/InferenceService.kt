package com.zithexagon.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.zithexagon.JobUi
import com.zithexagon.MainActivity
import com.zithexagon.R
import com.zithexagon.ZitDiag
import com.zithexagon.engine.EngineState
import com.zithexagon.engine.Stage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch

/**
 * Foreground service ("inferencia") that owns the lifetime of the process-wide engine ([EngineHost]).
 * Liberar / Seguro-idle / onTrimMemory / app exit must [tearDownAndStop]: release native resources,
 * stopForeground(REMOVE), cancel NOTIF_ID (and legacy QNN ids), stopSelf.
 */
class InferenceService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var watcher: Job? = null
    private var wake: PowerManager.WakeLock? = null
    private var foreground = false
    private var sawLoaded = false
    private var tearingDown = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel(this)
        purgeLegacyChannels(this)
        running = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_RELEASE, ACTION_STOP -> {
                // Do NOT goForeground+notify first (that re-posts the ongoing notif). Satisfy the
                // FGS contract only if we were just startForegroundService'd, then tear down sync:
                // unload kicks off async; stopForeground(REMOVE) + cancel(NOTIF_ID) + stopSelf now.
                ZitDiag.bc("notification: release (${intent.getStringExtra(EXTRA_WHY) ?: intent.action})")
                if (!foreground) {
                    runCatching {
                        ServiceCompat.startForeground(
                            this, NOTIF_ID,
                            build(null, null, EngineState.Idle),
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
                        )
                        foreground = true
                    }
                }
                EngineHost.release(this, stopService = false)
                tearDownAndStop("action=${intent.action}")
                return START_NOT_STICKY
            }
            ACTION_CANCEL -> {
                // Still need a valid FGS after startForegroundService
                goForeground(build(EngineHost.job.value, EngineHost.lastOutcome.value))
                ZitDiag.bc("notification: cancel")
                EngineHost.cancel()
            }
            else -> goForeground(build(EngineHost.job.value, EngineHost.lastOutcome.value))
        }
        if (watcher == null && !tearingDown) watcher = scope.launch {
            val engine = EngineHost.engine(this@InferenceService)
            combine(EngineHost.job, EngineHost.lastOutcome, engine.state) { j, o, s -> Triple(j, o, s) }
                .distinctUntilChanged()
                .collect { (j, o, s) ->
                    if (tearingDown) return@collect
                    setWake(j != null)
                    if (s is EngineState.Ready) sawLoaded = true
                    // Engine fully released (Liberar elsewhere, Seguro freed to Idle, trimMemory): drop FGS.
                    // Keep a one-shot "Imagen lista" auto-cancel notif if we just finished.
                    if (j == null && s is EngineState.Idle && !EngineHost.isLoaded && sawLoaded) {
                        val done = o as? EngineHost.Outcome.Done
                        if (done != null) postDoneAutoCancel(done)
                        tearDownAndStop("idle-released")
                        return@collect
                    }
                    notify(build(j, o, s))
                }
        }
        return START_NOT_STICKY
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        ZitDiag.bc("service timeout type=$fgsType")
        EngineHost.release(this, stopService = false)
        tearDownAndStop("timeout")
    }

    override fun onDestroy() {
        tearingDown = true
        setWake(false)
        scope.cancel()
        running = false
        runCatching { ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE) }
        clearNotification(this)
        super.onDestroy()
    }

    private fun tearDownAndStop(why: String) {
        if (tearingDown) {
            stopSelf()
            return
        }
        tearingDown = true
        ZitDiag.bc("service tearDown ($why)")
        watcher?.cancel(); watcher = null
        setWake(false)
        runCatching { ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE) }
        foreground = false
        clearNotification(this)
        stopSelf()
    }

    private fun goForeground(n: Notification) {
        if (tearingDown) return
        if (foreground) { notify(n); return }
        runCatching {
            ServiceCompat.startForeground(this, NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            foreground = true
        }.onFailure { ZitDiag.bc("startForeground failed: $it"); stopSelf() }
    }

    private fun notify(n: Notification) {
        if (tearingDown) return
        runCatching { getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n) }
    }

    /** Non-ongoing "Imagen lista" so the FGS can go away after Seguro/Liberar without losing the tap target. */
    private fun postDoneAutoCancel(done: EngineHost.Outcome.Done) {
        val open = PendingIntent.getActivity(
            this, 11,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                .putExtra(MainActivity.EXTRA_OPEN_ITEM, done.itemId),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val n = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(getString(R.string.notif_done))
            .setContentText(getString(R.string.notif_tap_open))
            .setContentIntent(open)
            .setAutoCancel(true)
            .setOngoing(false)
            .setOnlyAlertOnce(true)
            .setWhen(done.at).setShowWhen(true)
            .build()
        runCatching { getSystemService(NotificationManager::class.java).notify(DONE_NOTIF_ID, n) }
    }

    private fun setWake(on: Boolean) {
        if (on && wake?.isHeld != true) {
            wake = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "zithexagon:inferencia").apply {
                    setReferenceCounted(false); acquire(60 * 60 * 1000L)
                }
        } else if (!on) {
            wake?.let { if (it.isHeld) runCatching { it.release() } }; wake = null
        }
    }

    private fun build(j: JobUi?, o: EngineHost.Outcome?, s: EngineState? = null): Notification {
        val open = PendingIntent.getActivity(
            this, 1,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                .apply { if (j == null && o is EngineHost.Outcome.Done) putExtra(MainActivity.EXTRA_OPEN_ITEM, o.itemId) },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val release = PendingIntent.getService(
            this, 2, intent(this, ACTION_RELEASE).putExtra(EXTRA_WHY, "action"), PendingIntent.FLAG_IMMUTABLE,
        )
        val dismissed = PendingIntent.getService(
            this, 3, intent(this, ACTION_RELEASE).putExtra(EXTRA_WHY, "dismissed"), PendingIntent.FLAG_IMMUTABLE,
        )
        val b = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentIntent(open)
            .setDeleteIntent(dismissed)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setOngoing(true)
        if (j != null) {
            val text = when (j.stage) {
                Stage.LOADING -> getString(R.string.notif_loading) + (j.loadDetail?.let { " · $it" } ?: "")
                Stage.TEXT_ENCODER -> getString(R.string.notif_text)
                Stage.DENOISE -> if (j.total > 0) getString(R.string.notif_step, j.step, j.total) else getString(R.string.notif_denoise)
                Stage.VAE_DECODE -> getString(R.string.notif_vae)
                Stage.UPSCALE -> getString(R.string.notif_upscale)
            }
            b.setContentTitle(getString(if (j.kind == JobUi.Kind.UPSCALE) R.string.notif_title_upscale else R.string.notif_title_gen))
                .setContentText(text)
                .setProgress(1000, (j.fraction.coerceIn(0f, 1f) * 1000).toInt(), j.stage == Stage.LOADING && j.loadProgress <= 0f)
                .setWhen(j.startedAt).setShowWhen(true).setUsesChronometer(true)
                .addAction(0, getString(R.string.notif_cancel), PendingIntent.getService(this, 4, intent(this, ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE))
                .addAction(0, getString(R.string.notif_release), release)
        } else {
            val loaded = (s ?: EngineHost.engine(this).state.value) is EngineState.Ready
            val title = when (o) {
                is EngineHost.Outcome.Done -> getString(R.string.notif_done)
                is EngineHost.Outcome.Failed -> getString(R.string.notif_failed)
                is EngineHost.Outcome.Cancelled -> getString(R.string.notif_cancelled)
                null -> getString(R.string.notif_idle_title)
            }
            val text = (if (o is EngineHost.Outcome.Failed) o.message.take(120) + " · " else if (o is EngineHost.Outcome.Done) getString(R.string.notif_tap_open) + " · " else "") +
                getString(if (loaded) R.string.notif_model_loaded else R.string.notif_model_not_loaded)
            b.setContentTitle(title).setContentText(text)
                .setStyle(NotificationCompat.BigTextStyle().bigText(text))
                .setShowWhen(o != null).setUsesChronometer(false)
                .addAction(0, getString(R.string.notif_release), release)
            if (o != null) b.setWhen(
                when (o) {
                    is EngineHost.Outcome.Done -> o.at
                    is EngineHost.Outcome.Failed -> o.at
                    is EngineHost.Outcome.Cancelled -> o.at
                },
            )
        }
        return b.build()
    }

    companion object {
        const val CHANNEL = "inferencia"
        const val NOTIF_ID = 4242
        /** Auto-cancel "Imagen lista" after FGS is removed. */
        const val DONE_NOTIF_ID = 4243
        const val ACTION_START = "com.zithexagon.action.START"
        const val ACTION_CANCEL = "com.zithexagon.action.CANCEL"
        const val ACTION_RELEASE = "com.zithexagon.action.RELEASE"
        const val ACTION_STOP = "com.zithexagon.action.STOP"
        const val EXTRA_WHY = "why"
        @Volatile var running = false; private set

        /** Legacy QNN / older channel + notification ids that must not linger after Liberar. */
        private val LEGACY_CHANNELS = arrayOf("qnn", "qnn_inferencia", "inference", "zit_inferencia", "motor")
        private val LEGACY_NOTIF_IDS = intArrayOf(1, 42, 1001, 10001, 2001, 4242)

        fun intent(ctx: Context, action: String?) =
            Intent(ctx, InferenceService::class.java).also { if (action != null) it.action = action }

        fun ensureChannel(ctx: Context) {
            val nm = ctx.getSystemService(NotificationManager::class.java)
            if (nm.getNotificationChannel(CHANNEL) == null) nm.createNotificationChannel(
                NotificationChannel(CHANNEL, ctx.getString(R.string.notif_channel), NotificationManager.IMPORTANCE_LOW).apply {
                    description = ctx.getString(R.string.notif_channel_desc); setShowBadge(false)
                },
            )
        }

        fun purgeLegacyChannels(ctx: Context) {
            val nm = ctx.getSystemService(NotificationManager::class.java) ?: return
            for (id in LEGACY_CHANNELS) runCatching { nm.deleteNotificationChannel(id) }
        }

        /** Cancel FGS + legacy notification ids. Safe to call from UI / EngineHost after stopService. */
        fun clearNotification(ctx: Context) {
            val nm = ctx.applicationContext.getSystemService(NotificationManager::class.java) ?: return
            for (id in LEGACY_NOTIF_IDS) runCatching { nm.cancel(id) }
            runCatching { nm.cancel(NOTIF_ID) }
            purgeLegacyChannels(ctx.applicationContext)
            ZitDiag.bc("notification cleared (id=$NOTIF_ID + legacy)")
        }

        /** Ask the running service to tear down without re-posting FGS. Falls back to cancel+stopService. */
        fun stopAndClear(ctx: Context) {
            val app = ctx.applicationContext
            if (running) {
                runCatching { app.startService(intent(app, ACTION_STOP).putExtra(EXTRA_WHY, "stopAndClear")) }
                    .onFailure {
                        runCatching { app.stopService(intent(app, null)) }
                        clearNotification(app)
                    }
            } else {
                runCatching { app.stopService(intent(app, null)) }
                clearNotification(app)
            }
        }
    }
}
