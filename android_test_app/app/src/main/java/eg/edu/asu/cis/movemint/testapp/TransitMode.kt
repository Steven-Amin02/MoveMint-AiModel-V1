package eg.edu.asu.cis.movemint.testapp

enum class TransitCategory {
    STILL,
    MICRO_MOBILITY,
    PRIVATE_VEHICLE,
    PUBLIC_TRANSIT
}

enum class TransitMode(
    val index: Int,
    val displayName: String,
    val category: TransitCategory,
    val badgeColor: Int // ARGB hex
) {
    STILL(0, "Still", TransitCategory.STILL, 0xFF78909C.toInt()),
    WALK(1, "Walking", TransitCategory.MICRO_MOBILITY, 0xFF2E7D32.toInt()),
    RUN(2, "Running", TransitCategory.MICRO_MOBILITY, 0xFF00897B.toInt()),
    BIKE(3, "Cycling", TransitCategory.MICRO_MOBILITY, 0xFF43A047.toInt()),
    CAR(4, "Private Car", TransitCategory.PRIVATE_VEHICLE, 0xFFE53935.toInt()),
    BUS(5, "Bus", TransitCategory.PUBLIC_TRANSIT, 0xFF1E88E5.toInt()),
    TRAIN(6, "Train", TransitCategory.PUBLIC_TRANSIT, 0xFF3949AB.toInt()),
    SUBWAY(7, "Cairo Metro", TransitCategory.PUBLIC_TRANSIT, 0xFF8E24AA.toInt());

    val isPublicTransit: Boolean
        get() = category == TransitCategory.PUBLIC_TRANSIT

    val isMicroMobility: Boolean
        get() = category == TransitCategory.MICRO_MOBILITY

    val isPrivateVehicle: Boolean
        get() = category == TransitCategory.PRIVATE_VEHICLE

    companion object {
        fun fromIndex(idx: Int): TransitMode {
            return entries.find { it.index == idx } ?: STILL
        }
    }
}

data class ClassificationResult(
    val topMode: TransitMode,
    val confidence: Float,
    val probabilities: FloatArray,
    val isAntiCheatVerified: Boolean,
    val antiCheatMessage: String
)
