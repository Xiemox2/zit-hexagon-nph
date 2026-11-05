package com.zithexagon.data

/** Prompt editing for trigger-word chips: comma-separated phrases, no duplicates (from Hunyuan NPU). */
object PromptKeywords {
    private fun parts(prompt: String) = prompt.split(',').map { it.trim() }.filter { it.isNotEmpty() }
    fun contains(prompt: String, kw: String) = parts(prompt).any { it.equals(kw.trim(), ignoreCase = true) }
    fun add(prompt: String, kw: String): String =
        if (contains(prompt, kw)) prompt else if (prompt.isBlank()) kw.trim() else prompt.trimEnd().trimEnd(',') + ", " + kw.trim()
    fun remove(prompt: String, kw: String): String = parts(prompt).filterNot { it.equals(kw.trim(), ignoreCase = true) }.joinToString(", ")
    fun toggle(prompt: String, kw: String) = if (contains(prompt, kw)) remove(prompt, kw) else add(prompt, kw)
}
