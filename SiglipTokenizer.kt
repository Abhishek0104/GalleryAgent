package com.example.siglipsemanticsearch

import android.content.Context
import android.util.JsonReader
import android.util.JsonToken
import android.util.Log
import java.text.Normalizer
import kotlin.math.max

/**
 * A true Unigram Tokenizer implementation for SigLIP.
 * * Unlike BPE, Unigram uses the Viterbi algorithm to find the most probable
 * segmentation of the text based on token scores (log probabilities).
 * * Requires a `tokenizer.json` file containing "model": { "type": "Unigram", "vocab": [[token, score], ...] }
 */
class SiglipUnigramTokenizer(context: Context, fileName: String = "tokenizer.json") {

    data class Token(val text: String, val score: Double, val id: Int)

    private val vocabMap = HashMap<String, Token>()
    private var maxTokenLength = 0
    private var bosTokenId = 0 // <bos> is usually 0 or defined in special tokens
    private var eosTokenId = 1 // <eos> is usually 1
    private var unkTokenId = 0

    // Standard SentencePiece underscore
    private val SPIECE_UNDERLINE = "\u2581" 

    init {
        loadTokenizerJson(context, fileName)
    }

    /**
     * Tokenizes text using the Unigram Viterbi algorithm.
     */
    fun tokenize(text: String): IntArray {
        // 1. Normalize (NFKC) and replace spaces
        // SigLIP usually lowercases and replaces spaces with U+2581
        var cleanText = Normalizer.normalize(text, Normalizer.Form.NFKC)
        
        // Ensure leading underscore if not empty
        cleanText = cleanText.replace(" ", SPIECE_UNDERLINE)
        if (!cleanText.startsWith(SPIECE_UNDERLINE)) {
            cleanText = SPIECE_UNDERLINE + cleanText
        }

        // 2. Viterbi Algorithm to find best segmentation
        val bestPath = viterbi(cleanText)

        // 3. Convert to IDs
        val ids = ArrayList<Int>()
        
        // Add BOS if needed (SigLIP typically does NOT use BOS for text input, check your model config)
        // ids.add(bosTokenId) 

        ids.addAll(bestPath.map { it.id })

        // Add EOS
        ids.add(eosTokenId)

        // 4. Pad/Truncate (Fixed to 64 for your SigLIP model)
        return padOrTruncate(ids, 64)
    }

    /**
     * The Viterbi algorithm finds the path with the highest total score.
     * DP[i] = max score to reach index i.
     */
    private fun viterbi(text: String): List<Token> {
        val n = text.length
        val bestScores = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val bestPaths = arrayOfNulls<Token>(n + 1)
        val fromIndex = IntArray(n + 1) { -1 }

        bestScores[0] = 0.0

        for (i in 0 until n) {
            if (bestScores[i] == Double.NEGATIVE_INFINITY) continue

            // Try all possible substrings starting at i
            // Optimization: Don't check beyond maxTokenLength
            val limit = kotlin.math.min(n, i + maxTokenLength)
            
            for (j in i + 1..limit) {
                val sub = text.substring(i, j)
                val token = vocabMap[sub]

                if (token != null) {
                    val newScore = bestScores[i] + token.score
                    if (newScore > bestScores[j]) {
                        bestScores[j] = newScore
                        bestPaths[j] = token
                        fromIndex[j] = i
                    }
                }
            }
            
            // Handle Unknown Characters (Fallback)
            // If we can't move forward and we are at a single character that isn't in vocab
            if (bestScores[i+1] == Double.NEGATIVE_INFINITY) {
                 // In a rigorous implementation, we might back off to bytes.
                 // Here we assume the model has <unk> or individual chars.
                 // If a char is strictly unknown, we skip or use UNK logic.
            }
        }

        // Backtrack to reconstruct path
        val result = java.util.LinkedList<Token>()
        var curr = n
        
        if (bestScores[n] == Double.NEGATIVE_INFINITY) {
            Log.e("Unigram", "Failed to tokenize string completely. Fallback to UNK.")
            return listOf(vocabMap["<unk>"] ?: Token("<unk>", 0.0, unkTokenId))
        }

        while (curr > 0) {
            val token = bestPaths[curr]!!
            result.addFirst(token)
            curr = fromIndex[curr]
        }

        return result
    }

    private fun padOrTruncate(tokens: List<Int>, maxLength: Int): IntArray {
        if (tokens.size > maxLength) {
            // Keep EOS? Usually for SigLIP simply cutting off is acceptable, 
            // but keeping the last token as EOS is safer.
            val truncated = tokens.take(maxLength - 1).toMutableList()
            truncated.add(eosTokenId)
            return truncated.toIntArray()
        }
        
        val result = IntArray(maxLength)
        for (i in tokens.indices) {
            result[i] = tokens[i]
        }
        // Remaining are 0 (pad)
        return result
    }

    private fun loadTokenizerJson(context: Context, fileName: String) {
        try {
            context.assets.open(fileName).use { inputStream ->
                val reader = JsonReader(inputStream.reader())
                parseJson(reader)
            }
            Log.d("SiglipTokenizer", "Loaded Unigram Vocab: ${vocabMap.size} tokens. Max Len: $maxTokenLength")
        } catch (e: Exception) {
            Log.e("SiglipTokenizer", "Error loading tokenizer.json", e)
            // Fallback for empty init
            vocabMap["<unk>"] = Token("<unk>", -100.0, 0)
        }
    }

    /**
     * Parses generic HuggingFace tokenizer.json structure
     */
    private fun parseJson(reader: JsonReader) {
        reader.beginObject()
        while (reader.hasNext()) {
            val name = reader.nextName()
            when (name) {
                "model" -> parseModel(reader)
                "added_tokens" -> parseAddedTokens(reader) // For special tokens like <bos>, <eos>
                else -> reader.skipValue()
            }
        }
        reader.endObject()
    }

    private fun parseModel(reader: JsonReader) {
        reader.beginObject()
        while (reader.hasNext()) {
            val name = reader.nextName()
            when (name) {
                "vocab" -> parseVocabArray(reader)
                else -> reader.skipValue()
            }
        }
        reader.endObject()
    }

    /**
     * Unigram vocab is an ARRAY of [token, score]
     * Example: [ ["<unk>", 0.0], [" the", -1.4], ... ]
     */
    private fun parseVocabArray(reader: JsonReader) {
        reader.beginArray()
        var index = 0
        while (reader.hasNext()) {
            reader.beginArray()
            if (reader.hasNext()) {
                val tokenText = reader.nextString()
                val score = if (reader.hasNext()) reader.nextDouble() else 0.0
                
                // Track max length for Viterbi optimization
                if (tokenText.length > maxTokenLength) maxTokenLength = tokenText.length
                
                vocabMap[tokenText] = Token(tokenText, score, index)
            }
            // Consume remaining items in the inner array if any
            while (reader.hasNext()) reader.skipValue()
            reader.endArray()
            index++
        }
        reader.endArray()
    }
    
    private fun parseAddedTokens(reader: JsonReader) {
        reader.beginArray()
        while(reader.hasNext()) {
            reader.beginObject()
            var content = ""
            var id = -1
            while(reader.hasNext()) {
                when(reader.nextName()) {
                    "content" -> content = reader.nextString()
                    "id" -> id = reader.nextInt()
                    else -> reader.skipValue()
                }
            }
            reader.endObject()
            
            // Assign special IDs
            if (content == "</s>" || content == "<eos>") eosTokenId = id
            if (content == "<s>" || content == "<bos>") bosTokenId = id
            if (content == "<unk>") unkTokenId = id
        }
        reader.endArray()
    }
}

