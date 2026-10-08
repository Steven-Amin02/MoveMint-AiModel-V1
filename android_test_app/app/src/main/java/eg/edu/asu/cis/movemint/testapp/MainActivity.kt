package eg.edu.asu.cis.movemint.testapp

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import eg.edu.asu.cis.movemint.testapp.databinding.ActivityMainBinding
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: TransitClassifier
    private lateinit var collector: SensorCollector

    private val progressBars = mutableMapOf<Int, ProgressBar>()
    private val percentTexts = mutableMapOf<Int, TextView>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 1. Initialize TFLite Classifier
        classifier = TransitClassifier(this)

        // 2. Build 8-Class Probability Rows
        setupProbabilityRows()

        // 3. Initialize Sensor Collector
        collector = SensorCollector(
            context = this,
            onSampleUpdate = { currentCount, maxCount, liveSample ->
                runOnUiThread {
                    binding.pbBuffer.progress = currentCount
                    binding.tvBufferCount.text = "$currentCount / $maxCount samples"
                    updateLiveTelemetryText(liveSample)
                }
            },
            onBurstComplete = { burstBuffer ->
                // Run on-device inference on background thread
                Thread {
                    val result = classifier.classifyBurst(burstBuffer)
                    runOnUiThread {
                        displayClassificationResult(result)
                    }
                }.start()
            }
        )

        // 4. Setup Controls
        binding.btnStartBurst.setOnClickListener {
            if (collector.isRunning.get()) {
                collector.stopCollection()
                binding.btnStartBurst.text = getString(R.string.btn_start_burst)
                binding.tvStatusTitle.text = "Sensor Status: STOPPED"
            } else {
                collector.startCollection(continuous = binding.switchLiveStream.isChecked)
                binding.btnStartBurst.text = getString(R.string.btn_stop_test)
                binding.tvStatusTitle.text = if (binding.switchLiveStream.isChecked) {
                    "Sensor Status: LIVE MONITORING (50 Hz)"
                } else {
                    "Sensor Status: RECORDING 9s BURST (50 Hz)"
                }
            }
        }

        binding.switchLiveStream.setOnCheckedChangeListener { _, isChecked ->
            if (collector.isRunning.get()) {
                collector.stopCollection()
                collector.startCollection(continuous = isChecked)
                binding.tvStatusTitle.text = if (isChecked) {
                    "Sensor Status: LIVE MONITORING (50 Hz)"
                } else {
                    "Sensor Status: RECORDING 9s BURST (50 Hz)"
                }
            }
        }
    }

    private fun setupProbabilityRows() {
        binding.containerProbabilities.removeAllViews()

        // Header Title
        val titleView = TextView(this).apply {
            text = "Mode Probability Distribution"
            textSize = 15f
            setTextColor(getColor(R.color.text_primary))
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, 0, 0, 20)
        }
        binding.containerProbabilities.addView(titleView)

        for (mode in TransitMode.entries) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 8, 0, 8) }
            }

            val label = TextView(this).apply {
                text = mode.displayName
                textSize = 13f
                setTextColor(getColor(R.color.text_primary))
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.3f)
            }

            val pb = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 100
                progress = 0
                progressTintList = android.content.res.ColorStateList.valueOf(mode.badgeColor)
                layoutParams = LinearLayout.LayoutParams(0, 24, 2f)
            }

            val percent = TextView(this).apply {
                text = "0.0%"
                textSize = 12f
                setTextColor(getColor(R.color.text_secondary))
                gravity = android.view.Gravity.END
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 0.9f)
            }

            row.addView(label)
            row.addView(pb)
            row.addView(percent)

            binding.containerProbabilities.addView(row)

            progressBars[mode.index] = pb
            percentTexts[mode.index] = percent
        }
    }

    private fun updateLiveTelemetryText(sample: FloatArray) {
        val accX = sample[0]
        val accY = sample[1]
        val accZ = sample[2]
        val gyrX = sample[6]
        val gyrY = sample[7]
        val gyrZ = sample[8]
        val magX = sample[9]
        val magY = sample[10]
        val magZ = sample[11]

        binding.tvRawSensors.text = String.format(
            Locale.US,
            "Acc:  [%+5.2f, %+5.2f, %+5.2f] m/s²\nGyr:  [%+5.2f, %+5.2f, %+5.2f] rad/s\nMag:  [%+5.2f, %+5.2f, %+5.2f] µT",
            accX, accY, accZ, gyrX, gyrY, gyrZ, magX, magY, magZ
        )
    }

    private fun displayClassificationResult(result: ClassificationResult) {
        val top = result.topMode
        binding.tvWinningMode.text = top.displayName
        binding.tvConfidence.text = String.format(Locale.US, "Confidence: %.1f%%", result.confidence * 100f)
        binding.tvVerdictMessage.text = result.antiCheatMessage

        // Update Badge
        binding.tvModeBadge.text = when {
            top.isPublicTransit -> "PUBLIC TRANSIT"
            top.isPrivateVehicle -> "PRIVATE CAR (ANTI-CHEAT)"
            top.isMicroMobility -> "MICRO-MOBILITY"
            else -> "STILL"
        }

        val badgeDrawable = binding.tvModeBadge.background as? GradientDrawable
            ?: GradientDrawable().apply { cornerRadius = 40f }
        badgeDrawable.setColor(top.badgeColor)
        binding.tvModeBadge.background = badgeDrawable

        // If burst mode finished, reset button text
        if (!binding.switchLiveStream.isChecked) {
            binding.btnStartBurst.text = getString(R.string.btn_start_burst)
            binding.tvStatusTitle.text = "Sensor Status: BURST COMPLETE"
        }

        // Update 8 Class Progress Bars
        for (i in 0 until TransitClassifier.NUM_CLASSES) {
            val prob = result.probabilities[i]
            val pct = (prob * 100f).toInt()
            progressBars[i]?.progress = pct
            percentTexts[i]?.text = String.format(Locale.US, "%.1f%%", prob * 100f)
        }
    }

    override fun onPause() {
        super.onPause()
        collector.stopCollection()
        binding.btnStartBurst.text = getString(R.string.btn_start_burst)
        binding.tvStatusTitle.text = "Sensor Status: PAUSED"
    }

    override fun onDestroy() {
        super.onDestroy()
        collector.stopCollection()
        classifier.close()
    }
}
