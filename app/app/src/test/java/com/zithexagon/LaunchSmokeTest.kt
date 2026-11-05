package com.zithexagon

import androidx.lifecycle.ViewModelProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** Regression test for the 0.23.1 launch crash: the real MainActivity must start with the default ViewModel factory. */
@RunWith(AndroidJUnit4::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35])
class LaunchSmokeTest {
    @Test fun appViewModelHasApplicationConstructor() {
        assertNotNull(AppViewModel::class.java.getConstructor(android.app.Application::class.java))
    }

    @Test fun mainActivityLaunchesWithoutStartupError() {
        val ctx = RuntimeEnvironment.getApplication()
        CrashReporter.clear(ctx)
        val act = Robolectric.buildActivity(MainActivity::class.java).setup().get()
        assertNotNull(ViewModelProvider(act)[AppViewModel::class.java])
        assertFalse("startup error recorded: " + CrashReporter.report(ctx), CrashReporter.hasUnseen(ctx))
    }
}
