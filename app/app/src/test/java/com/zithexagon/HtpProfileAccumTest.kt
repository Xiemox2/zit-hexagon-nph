package com.zithexagon

import com.zithexagon.engine.HtpProfileAccum
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Host-side path: exact profile-batch format from ggml_hexagon_dump_batch_prof
 * through the same regex/parser Kotlin uses at runtime (no DSP needed).
 */
class HtpProfileAccumTest {
    private val sample =
        "ggml-hexagon.cpp:252 - ggml-hex: HTP0 profile-batch n=42 dsp_us=5100000 host_wait_us=120000 " +
            "hmx_us=4000000(10) hvx_us=800000(20) dma_us=100000(5) flush_us=50000(2) fence_us=30000(3) other_us=120000(2)\n"

    @Test fun parsesExactFormatAndSelfCheck() {
        val a = HtpProfileAccum()
        a.ingest(sample)
        a.onSamplerStep(1, 6.5f)
        val rep = a.report(listOf(6.5f))
        assertTrue("self-check missing", rep.contains("perfil: activo en motor=sí"))
        assertTrue(rep.contains("líneas recibidas=1"))
        assertTrue(rep.contains("formato ejemplo="))
        assertTrue(rep.contains("batches 1"))
        assertTrue(rep.contains("HMX 4000.0 ms"))
        assertTrue(rep.contains("HVX 800.0 ms"))
        assertTrue(rep.contains("host-wait/RPC 120.0 ms"))
    }

    @Test fun emptyYieldsDiagFromLogTail() {
        val a = HtpProfileAccum()
        a.ingestLogTail(
            "util.cpp:628 - ggml-hex: HTP0 hwinfo: threads 4\n" +
                "util.cpp:628 - ggml-hex: HTP0 profiler opt_profile=0\n" +
                "other noise\n"
        )
        a.onSamplerStep(1, 7f)
        val rep = a.report(listOf(7f))
        assertTrue(rep.contains("líneas recibidas=0"))
        assertTrue(rep.contains("activo en motor=no"))
        assertTrue("diag dump missing", rep.contains("Diagnóstico"))
        assertTrue(rep.contains("opt_profile=0"))
    }

    @Test fun parsesLoraFields() {
        val a = HtpProfileAccum()
        a.ingest(sample.trimEnd() + " lora_mm_us=900000(240) lora_ew_us=150000(120)\n")
        a.onSamplerStep(1, 6.5f)
        val rep = a.report(listOf(6.5f))
        assertTrue(rep, rep.contains("LoRA (incluido arriba): matmul 900.0 ms (240) · elementwise 150.0 ms (120)"))
        assertTrue(rep.contains("TOTAL LoRA: matmul 900.0 ms (240)"))
        assertTrue(rep.contains("HMX 4000.0 ms"))
    }

    @Test fun parsesFusedLoraFields() {
        val a = HtpProfileAccum()
        a.ingest(sample.trimEnd() + " lora_mm_us=300000(60) lora_ew_us=50000(60) lora_fused_us=2500000(120) nop=120\n")
        a.onSamplerStep(1, 6.5f)
        val rep = a.report(listOf(6.5f))
        assertTrue(rep, rep.contains("LoRA (incluido arriba): matmul 300.0 ms (60) · elementwise 50.0 ms (60)"))
        assertTrue(rep, rep.contains("LoRA fusionada en HMX (base+LoRA, incluida en HMX): 2500.0 ms (120 ops) · no-ops 120"))
        assertTrue(rep, rep.contains("fusionada base+LoRA 2500.0 ms (120)"))
    }

    @Test fun regexMatchesPrefixedLogLine() {
        val m = HtpProfileAccum.RE.find(sample)
        assertTrue(m != null)
        assertEquals("42", m!!.groupValues[1])
        assertEquals("5100000", m.groupValues[2])
        assertEquals("120000", m.groupValues[3])
        assertEquals("4000000", m.groupValues[4])
    }

    @Test fun nonProfileLineIgnored() {
        val a = HtpProfileAccum()
        a.ingest("ggml-hex: HTP0 profile-op MUL_MAT|hmx|usec 100\n")
        assertFalse(a.report(emptyList()).contains("líneas recibidas=1"))
        assertTrue(a.report(emptyList()).contains("líneas recibidas=0"))
    }

    @Test fun parsesHostOverhead() {
        val a = HtpProfileAccum()
        a.ingest(
            "ggml_runner.cpp:640 - nph-host-overhead build_ms=312.5 alloc_ms=8421.0 " +
                "did_reserve=1 cache=MISS reserves=1 ws_mb=1536.0 keep=1\n"
        )
        a.ingest(
            "ggml_runner.cpp:640 - nph-host-overhead build_ms=280.0 alloc_ms=12.0 " +
                "did_reserve=0 cache=HIT reserves=1 ws_mb=1536.0 keep=1\n"
        )
        val h = a.hostReport()
        assertTrue(h.contains("paso-1 host:"))
        assertTrue(h.contains("alloc 8421 ms"))
        assertTrue(h.contains("cache=MISS"))
        assertTrue(h.contains("compute-2:"))
        assertTrue(h.contains("cache=HIT"))
        val m = HtpProfileAccum.HOST_RE.find(
            "nph-host-overhead build_ms=1.0 alloc_ms=2.0 did_reserve=0 cache=HIT reserves=1 ws_mb=3.0 keep=1"
        )
        assertTrue(m != null)
        assertEquals("HIT", m!!.groupValues[4])
    }

    @Test fun hostReportDitOnlyAndResidentGen() {
        val a = HtpProfileAccum()
        a.ingest("nph-host-overhead build_ms=50.0 alloc_ms=900.0 did_reserve=1 cache=MISS reserves=1 ws_mb=10.0 keep=0 runner=qwen3\n")
        a.ingest("nph-host-overhead build_ms=300.0 alloc_ms=15.0 did_reserve=0 cache=HIT reserves=1 ws_mb=1536.0 keep=1 runner=z_image\n")
        a.ingest("nph-host-overhead build_ms=280.0 alloc_ms=11.0 did_reserve=0 cache=HIT reserves=1 ws_mb=1536.0 keep=1 runner=z_image\n")
        a.ingest("nph-host-overhead build_ms=20.0 alloc_ms=400.0 did_reserve=1 cache=MISS reserves=1 ws_mb=800.0 keep=0 runner=vae\n")
        val h = a.hostReport(genOnCtx = 2)
        assertTrue(h, h.contains("gen #2 en contexto residente · compute-1: build 300 ms · alloc 15 ms · cache=HIT"))
        assertTrue(h, h.contains("computes DiT: 2 (HIT 2 · MISS 0)"))
        assertTrue(h, h.contains("qwen3×1"))
        assertTrue(h, h.contains("vae×1"))
        assertFalse(h, h.contains("alloc 900 ms · cache"))
    }

    @Test fun loraReportFromEngineLines() {
        val a = HtpProfileAccum()
        assertEquals("", a.loraReport())
        a.ingest("lora_combined.hpp:300 - nph-lora-file a.safetensors rank=16 dtype=bf16 strength=0.80\n")
        a.ingest("nph-lora-combined targets=240 rank_total=32 rank_padded=32 matmuls_per_dit_pass=480 (per-LoRA path: 960)\n")
        a.ingestLogTail("x nph-lora-file dup.safetensors\n")  // live lines present → log tail not re-ingested
        val r = a.loraReport()
        assertTrue(r, r.contains("LoRA (0.30"))
        assertTrue(r, r.contains("nph-lora-file a.safetensors rank=16"))
        assertTrue(r, r.contains("rank_padded=32"))
        assertFalse(r, r.contains("dup.safetensors"))
    }
}
