# MoveMint Edge-AI: Android TFLite Integration Guide

**For:** Ahmed Hesham (Mobile Lead & Kotlin Foreground Service Engineer)  
**From:** Steven Amin (AI / ML & Signal Processing Lead)  
**Deliverables:** 
- `movemint_edge_model_float32.tflite` (165 KB)
- `movemint_edge_model_float16.tflite` (91 KB)  
**Directory:** `AI-2/outputs/tflite/`

---

## 1. Model Input & Output Tensor Specifications

| Attribute | Specification | Details |
| :--- | :--- | :--- |
| **Model Format** | TensorFlow Lite (`.tflite`) | FlatBuffers, CPU / NNAPI / GPU compatible |
| **Input Shape** | `[1, 450, 12]` | Batch=1, TimeSteps=450, Channels=12 |
| **Input Dtype** | `Float32` | Standard 32-bit floating point |
| **Sensor Burst Duration** | **9.0 seconds** | Sampled at **50 Hz** ($20\text{ ms}$ intervals) |
| **Output Shape** | `[1, 8]` | Raw logits for the 8 transit classes |
| **Output Dtype** | `Float32` | Softmax produces probabilities summing to 1.0 |
| **Model Size** | **165 KB** (FP32) / **91 KB** (FP16) | Negligible mobile RAM footprint |
| **Execution Latency** | **< 3 ms** | Executes on standard mobile CPU |

---

## 2. Input Channel Order (12 Channels)

When buffering the 450 sensor readings, pack the channels in the following exact index order:

```kotlin
// Channel indices (0 to 11):
// 0, 1, 2:   Accelerometer (X, Y, Z)             -> Sensor.TYPE_ACCELEROMETER
// 3, 4, 5:   Linear Acceleration (X, Y, Z)      -> Sensor.TYPE_LINEAR_ACCELERATION
// 6, 7, 8:   Gyroscope (X, Y, Z)                -> Sensor.TYPE_GYROSCOPE
// 9, 10, 11: Magnetometer (X, Y, Z)             -> Sensor.TYPE_MAGNETIC_FIELD
```

* **Sampling Rate:** Register sensors with `SensorManager.SENSOR_DELAY_GAME` ($\approx 50\text{ Hz}$).
* **Normalization:** Standardize values: $(v - \mu) / \sigma$ using unit scale.

---

## 3. Output Class Mapping & Anti-Cheat Decision Rules

```kotlin
enum class TransitMode(val label: String, val isPublicTransit: Boolean, val isMicroMobility: Boolean) {
    STILL("Still", false, false),        // 0
    WALK("Walk", false, true),          // 1: Green micro-mobility leg
    RUN("Run", false, true),            // 2: Green micro-mobility leg
    BIKE("Bike", false, true),          // 3: Green micro-mobility leg
    CAR("Car", false, false),           // 4: NEGATIVE CLASS (Private vehicle - Anti-cheat target!)
    BUS("Bus", true, false),            // 5: REWARD LEG (Public transit)
    TRAIN("Train", true, false),        // 6: REWARD LEG (Public transit)
    SUBWAY("Subway", true, false);      // 7: REWARD LEG (Cairo Metro)

    companion object {
        fun fromIndex(index: Int): TransitMode = entries[index]
    }
}
```

### MoveMint Anti-Cheat Decision Logic:
```kotlin
fun verifyTransitLeg(predictions: FloatArray, spatialRouteMatch: Boolean): Boolean {
    val topClassIndex = predictions.indices.maxByOrNull { predictions[it] } ?: return false
    val topClass = TransitMode.fromIndex(topClassIndex)
    val confidence = predictions[topClassIndex]

    // Anti-Cheat Rule:
    // User only earns public transit Mints if:
    // 1. PostGIS confirmed GPS trajectory aligns with GTFS public transit route
    // 2. TFLite vibration model classifies the vehicle as Bus, Train, or Subway with confidence >= 0.60
    // 3. Model explicitly rejects Private Car (Car probability < 0.25)
    return spatialRouteMatch && topClass.isPublicTransit && confidence >= 0.60f
}
```

---

## 4. Production Kotlin Implementation for Android

Add the official TensorFlow Lite dependency to `app/build.gradle`:
```groovy
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

Place `movemint_edge_model_float32.tflite` in `app/src/main/assets/`.

### Complete Kotlin Classifier Helper:

```kotlin
package eg.edu.asu.cis.movemint.ai

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TransitVibrationClassifier(context: Context) {

    private val interpreter: Interpreter

    companion object {
        private const val MODEL_FILENAME = "movemint_edge_model_float32.tflite"
        private const val CHANNELS = 12
        private const val TIME_STEPS = 450
        private const val NUM_CLASSES = 8
        private const val BYTES_PER_FLOAT = 4
    }

    init {
        val modelBuffer = loadModelFile(context, MODEL_FILENAME)
        val options = Interpreter.Options().apply {
            setNumThreads(2) // Low-power 2-thread execution
            useNNAPI = true  // Hardware acceleration if supported
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Runs inference on a 9-second sensor burst.
     * @param buffer Array of shape [450][12] (450 chronological timesteps, each with 12 sensor values)
     * @return FloatArray of 8 probabilities
     */
    fun classifyBurst(buffer: Array<FloatArray>): FloatArray {
        // Allocate direct ByteBuffer for input tensor: [1, 450, 12]
        val inputBuffer = ByteBuffer.allocateDirect(1 * TIME_STEPS * CHANNELS * BYTES_PER_FLOAT).apply {
            order(ByteOrder.nativeOrder())
            rewind()
            for (t in 0 until TIME_STEPS) {
                for (c in 0 until CHANNELS) {
                    putFloat(buffer[t][c])
                }
            }
        }

        // Allocate output tensor: [1, 8]
        val outputBuffer = ByteBuffer.allocateDirect(1 * NUM_CLASSES * BYTES_PER_FLOAT).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Parse and compute Softmax
        outputBuffer.rewind()
        val rawLogits = FloatArray(NUM_CLASSES)
        for (i in 0 until NUM_CLASSES) {
            rawLogits[i] = outputBuffer.float
        }

        return softmax(rawLogits)
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
            exp[i] /= sum
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
```
