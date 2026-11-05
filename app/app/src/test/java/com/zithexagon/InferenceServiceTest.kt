package com.zithexagon

import android.app.NotificationManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.zithexagon.service.EngineHost
import com.zithexagon.service.InferenceService
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowLooper

/** 0.26.1: FGS starts with notification; Liberar unloads, stopForeground(REMOVE), cancels id, stopSelf. */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [35])
class InferenceServiceTest {
    @Test fun startsForegroundAndStopsOnRelease() {
        val ctx = RuntimeEnvironment.getApplication()
        val ctl = Robolectric.buildService(InferenceService::class.java, InferenceService.intent(ctx, InferenceService.ACTION_START))
        val svc = ctl.create().startCommand(0, 1).get()
        ShadowLooper.idleMainLooper()
        assertTrue(InferenceService.running)
        val n = shadowOf(svc).lastForegroundNotification
        assertNotNull("no foreground notification", n)
        assertEquals(InferenceService.CHANNEL, n.channelId)
        assertNotNull(ctx.getSystemService(NotificationManager::class.java).getNotificationChannel(InferenceService.CHANNEL))
        assertTrue("Liberar action missing", n.actions.any { it.title == ctx.getString(R.string.notif_release) })
        assertNotNull("swipe-away must release", n.deleteIntent)

        // Liberar → unload + stopForeground(REMOVE) + cancel(NOTIF_ID) + stopSelf
        ctl.withIntent(InferenceService.intent(ctx, InferenceService.ACTION_RELEASE)).startCommand(0, 2)
        ShadowLooper.idleMainLooper()

        assertTrue("service did not stopSelf after Liberar", shadowOf(svc).isStoppedBySelf || !InferenceService.running)
        assertNull(EngineHost.job.value)
        val nmShadow = shadowOf(ctx.getSystemService(NotificationManager::class.java))
        assertNull(
            "FGS notif id ${InferenceService.NOTIF_ID} still posted",
            nmShadow.getNotification(InferenceService.NOTIF_ID),
        )
        ctl.destroy()
        assertFalse(InferenceService.running)
    }
}
