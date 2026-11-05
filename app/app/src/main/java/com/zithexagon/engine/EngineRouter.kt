package com.zithexagon.engine

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Routes load/generate to [NphEngine] for NPH models (id `nph-*` or [LoadRequest.nph]
 * set) and to [FakeEngine] only for the explicit demo model.
 *
 * Never silently falls back to the mock engine for an NPH selection — if NPH
 * cannot load, [NphEngine.load] throws a Spanish error the UI surfaces.
 */
class EngineRouter(private val fake: ZitEngine, private val nph: ZitEngine) : ZitEngine {
    @Volatile private var cur: ZitEngine = fake
    private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
    override val state: StateFlow<EngineState> = _state.asStateFlow()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var mirror: Job? = null
    override val info: EngineInfo get() = cur.info

    init { follow(fake) }

    private fun follow(e: ZitEngine) {
        mirror?.cancel()
        mirror = scope.launch { e.state.collect { _state.value = it } }
    }

    private fun pick(request: LoadRequest): ZitEngine {
        val isNph = request.nph != null || request.modelId.startsWith("nph-")
        val isDemo = request.modelId.startsWith("demo")
        return when {
            isNph -> nph
            isDemo -> fake
            // Legacy folder packages (none ship anymore) — still try NPH rather than silent demo
            request.modelDir != null -> nph
            else -> fake
        }
    }

    override suspend fun load(request: LoadRequest) {
        val next = pick(request)
        if (next !== cur) {
            cur.unload()
            cur = next
            follow(next)
        }
        next.load(request)
    }

    override suspend fun unload() = cur.unload()
    override fun generate(request: GenerationRequest): Flow<GenerationEvent> = cur.generate(request)
    override fun upscale(request: UpscaleRequest): Flow<GenerationEvent> = cur.upscale(request)
    override fun cancel() = cur.cancel()

    fun nphEngine(): NphEngine? = nph as? NphEngine

    /** UI badge: honor the *selected* model, not whichever engine last ran. */
    fun infoFor(modelId: String?): EngineInfo =
        if (modelId != null && modelId.startsWith("nph-")) nph.info else cur.info
}
