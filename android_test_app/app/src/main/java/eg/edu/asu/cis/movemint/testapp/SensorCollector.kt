package eg.edu.asu.cis.movemint.testapp

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.atomic.AtomicBoolean

class SensorCollector(
    context: Context,
    private val onSampleUpdate: (collectedCount: Int, maxCount: Int, currentVector: FloatArray) -> Unit,
    private val onBurstComplete: (burstBuffer: Array<FloatArray>) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val linearAccelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gyroscope: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    // Current latest sensor readings (12 channels)
    private val latestAcc = FloatArray(3)
    private val latestLAcc = FloatArray(3)
    private val latestGyr = FloatArray(3)
    private val latestMag = FloatArray(3)

    // Gravity estimation fallback if TYPE_LINEAR_ACCELERATION is missing on device
    private val gravityFilter = FloatArray(3)
    private val hasHardwareLinearAcc = linearAccelerometer != null

    // Ring Buffer storage: 450 samples of 12 channels
    private val targetWindowSize = TransitClassifier.TIME_STEPS
    private val buffer = Array(targetWindowSize) { FloatArray(TransitClassifier.CHANNELS) }
    private var sampleCount = 0

    val isRunning = AtomicBoolean(false)
    private var isContinuousMode = false

    fun startCollection(continuous: Boolean = false) {
        if (isRunning.get()) return
        isRunning.set(true)
        isContinuousMode = continuous
        sampleCount = 0

        // Register sensors at 50 Hz (SENSOR_DELAY_GAME)
        accelerometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        linearAccelerometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroscope?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        magnetometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stopCollection() {
        if (!isRunning.get()) return
        isRunning.set(false)
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRunning.get()) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, latestAcc, 0, 3)

                // Software fallback for linear acceleration if hardware sensor is absent
                if (!hasHardwareLinearAcc) {
                    val alpha = 0.8f
                    gravityFilter[0] = alpha * gravityFilter[0] + (1 - alpha) * latestAcc[0]
                    gravityFilter[1] = alpha * gravityFilter[1] + (1 - alpha) * latestAcc[1]
                    gravityFilter[2] = alpha * gravityFilter[2] + (1 - alpha) * latestAcc[2]

                    latestLAcc[0] = latestAcc[0] - gravityFilter[0]
                    latestLAcc[1] = latestAcc[1] - gravityFilter[1]
                    latestLAcc[2] = latestAcc[2] - gravityFilter[2]
                }

                // Pack combined 12-channel vector on primary accelerometer tick
                packSample()
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                System.arraycopy(event.values, 0, latestLAcc, 0, 3)
            }
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, latestGyr, 0, 3)
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(event.values, 0, latestMag, 0, 3)
            }
        }
    }

    private fun packSample() {
        // Construct standard 12-channel frame matching MoveMint schema
        val sample = FloatArray(TransitClassifier.CHANNELS).apply {
            this[0] = latestAcc[0]
            this[1] = latestAcc[1]
            this[2] = latestAcc[2]
            this[3] = latestLAcc[0]
            this[4] = latestLAcc[1]
            this[5] = latestLAcc[2]
            this[6] = latestGyr[0]
            this[7] = latestGyr[1]
            this[8] = latestGyr[2]
            this[9] = latestMag[0]
            this[10] = latestMag[1]
            this[11] = latestMag[2]
        }

        if (sampleCount < targetWindowSize) {
            // Fill initial buffer
            buffer[sampleCount] = sample
            sampleCount++
            onSampleUpdate(sampleCount, targetWindowSize, sample)

            if (sampleCount == targetWindowSize) {
                // Initial 450-sample burst full!
                val snapshot = Array(targetWindowSize) { buffer[it].clone() }
                onBurstComplete(snapshot)

                if (!isContinuousMode) {
                    stopCollection()
                }
            }
        } else if (isContinuousMode) {
            // Shift sliding window (FIFO)
            for (i in 0 until targetWindowSize - 1) {
                buffer[i] = buffer[i + 1]
            }
            buffer[targetWindowSize - 1] = sample
            onSampleUpdate(targetWindowSize, targetWindowSize, sample)

            // Trigger real-time inference periodically (every ~50 samples = 1 sec)
            sampleCount++
            if (sampleCount % 50 == 0) {
                val snapshot = Array(targetWindowSize) { buffer[it].clone() }
                onBurstComplete(snapshot)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
