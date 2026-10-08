package eg.edu.asu.cis.movemint.testapp

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TransitClassifier(context: Context) {

    private val interpreter: Interpreter

    companion object {
        const val MODEL_FILENAME = "movemint_edge_model_float32.tflite"
        const val TIME_STEPS = 450
        const val CHANNELS = 12
        const val NUM_CLASSES = 8
        private const val BYTES_PER_FLOAT = 4
    }

    init {
        val modelBuffer = loadModelFile(context, MODEL_FILENAME)
        val options = Interpreter.Options().apply {
            setNumThreads(2) // Low-power background thread budget
            useNNAPI = false // CPU baseline for universal compatibility
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Executes edge inference on 450 sensor readings across 12 channels.
     * @param buffer Array of shape [450][12] (Chronological samples)
     * @return ClassificationResult containing top class, probabilities, and MoveMint anti-cheat verdict
     */
    fun classifyBurst(buffer: Array<FloatArray>): ClassificationResult {
        // Direct ByteBuffer matching TFLite Input [1, 450, 12]
        val inputBuffer = ByteBuffer.allocateDirect(1 * TIME_STEPS * CHANNELS * BYTES_PER_FLOAT).apply {
            order(ByteOrder.nativeOrder())
            rewind()
            for (t in 0 until TIME_STEPS) {
                for (c in 0 until CHANNELS) {
                    putFloat(buffer[t][c])
                }
            }
        }

        // Direct ByteBuffer matching TFLite Output [1, 8]
        val outputBuffer = ByteBuffer.allocateDirect(1 * NUM_CLASSES * BYTES_PER_FLOAT).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        // Execute on-device inference
        interpreter.run(inputBuffer, outputBuffer)

        // Read logits
        outputBuffer.rewind()
        val rawLogits = FloatArray(NUM_CLASSES)
        for (i in 0 until NUM_CLASSES) {
            rawLogits[i] = outputBuffer.float
        }

        // Apply Softmax
        val probabilities = softmax(rawLogits)

        // Find top class
        var topIndex = 0
        var maxProb = probabilities[0]
        for (i in 1 until NUM_CLASSES) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                topIndex = i
            }
        }

        val topMode = TransitMode.fromIndex(topIndex)

        // MoveMint Anti-Cheat Security Verdict
        val isVerified: Boolean
        val message: String

        when {
            topMode.isPublicTransit && maxProb >= 0.55f -> {
                isVerified = true
                message = "Public Transit Confirmed (${topMode.displayName}) - Points Awarded!"
            }
            topMode.isPrivateVehicle -> {
                isVerified = false
                message = "Private Car Detected - Anti-Cheat Flag (Transit Points Denied)"
            }
            topMode.isMicroMobility -> {
                isVerified = true
                message = "Active Eco-Commute (${topMode.displayName}) - Micro-Mobility Points!"
            }
            else -> {
                isVerified = false
                message = "Stationary / Ambiguous Motion - Telemetry Idle"
            }
        }

        return ClassificationResult(
            topMode = topMode,
            confidence = maxProb,
            probabilities = probabilities,
            isAntiCheatVerified = isVerified,
            antiCheatMessage = message
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        var sum = 0f
        val exp = FloatArray(logits.size)
        for (i in logits.indices) {
            exp[i] = kotlin.math.exp(logits[i] - maxLogit)
            sum += exp[i]
        }
        for (i in logits.indices) {
            exp[i] /= (sum + 1e-9f)
        }
        return exp
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun close() {
        interpreter.close()
    }
}
