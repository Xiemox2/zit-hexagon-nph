package com.zithexagon.service

import android.content.Context
import androidx.core.content.ContextCompat
import com.zithexagon.JobUi
import com.zithexagon.ZitDiag
import com.zithexagon.engine.EngineRouter
import com.zithexagon.engine.EngineState
import com.zithexagon.engine.FakeEngine
import com.zithexagon.engine.NphEngine
import com.zithexagon.engine.ZitEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

/**
 * The ONE engine of the process. 0.26: NPH (sd.cpp Hexagon). Lifetime tied to [InferenceService].
 */
object EngineHost {
    @Volatile private var _engine: ZitEngine? = null
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    val job = MutableStateFlow<JobUi?>(null)
    val lastOutcome = MutableStateFlow<Outcome?>(null)
    @Volatile var running: Job? = null
    private val lifecycle = Mutex()

    sealed interface Outcome {
        data class Done(val itemId: String, val at: Long = System.currentTimeMillis()) : Outcome
        data class Failed(val message: String, val at: Long = System.currentTimeMillis()) : Outcome
        data class Cancelled(val at: Long = System.currentTimeMillis()) : Outcome
    }

    fun engine(ctx: Context): ZitEngine = _engine ?: synchronized(this) {
        _engine ?: EngineRouter(FakeEngine(), NphEngine(ctx.applicationContext)).also { _engine = it }
    }

    fun install(e: ZitEngine) { synchronized(this) { _engine = e } }

    val isLoaded: Boolean get() = _engine?.state?.value.let { it is EngineState.Ready || it is EngineState.Busy || it is EngineState.Loading }

    fun ensureService(ctx: Context) {
        runCatching { ContextCompat.startForegroundService(ctx, InferenceService.intent(ctx, InferenceService.ACTION_START)) }
            .onFailure { ZitDiag.bc("service start failed: $it") }
    }

    fun cancel() {
        _engine?.cancel()
        val j = job.value ?: return
        if (j.stage == com.zithexagon.engine.Stage.LOADING) {
            running?.cancel(); lastOutcome.value = Outcome.Cancelled()
        }
    }

    fun onTrimMemory(level: Int, ctx: Context? = null) {
        ((_engine as? EngineRouter)?.nphEngine() ?: _engine as? NphEngine)?.onTrimMemory(level)
        // If the engine dropped to Idle, tear down the FGS notification too.
        val st = _engine?.state?.value
        if (ctx != null && st is EngineState.Idle && !isLoaded) {
            ZitDiag.bc("onTrimMemory → stopAndClear (level=$level)")
            InferenceService.stopAndClear(ctx)
        }
    }

    fun release(ctx: Context, stopService: Boolean = true): Job = scope.launch {
        lifecycle.withLock {
            ZitDiag.bc("release requested (job=${job.value?.kind})")
            val r = running
            if (r != null && r.isActive) {
                _engine?.cancel()
                if (withTimeoutOrNull(120_000) { r.join() } == null) { r.cancel(); r.join() }
            }
            runCatching { _engine?.unload() }.onFailure { ZitDiag.bc("unload failed: $it") }
            job.value = null
            lastOutcome.value = null
            ZitDiag.bc("released: NPH context freed")
        }
        if (stopService) InferenceService.stopAndClear(ctx.applicationContext)
    }
}
