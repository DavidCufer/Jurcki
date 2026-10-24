package si.jurcki.app.engine

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.abs

/** Loads the model package from `src/main/assets/model` and the parity vectors from the test resources. */
object TestData {

    fun resourceText(path: String): String {
        val stream = TestData::class.java.classLoader!!.getResourceAsStream(path)
            ?: throw IllegalStateException("test resource not found: $path")
        return stream.use { String(it.readBytes(), Charsets.UTF_8) }
    }

    fun json(path: String): JSONObject = JSONObject(resourceText(path))

    /** `app/src/main/assets/model`, found from the working directory (Gradle: the module dir `app/`). */
    val assetDir: File by lazy {
        val candidates = listOf("src/main/assets/model", "app/src/main/assets/model", "android/app/src/main/assets/model")
        candidates.map { File(it) }.firstOrNull { File(it, "model.json").isFile }
            ?: throw IllegalStateException("assets/model not found from ${File(".").absoluteFile}")
    }

    fun assetText(name: String): String = File(assetDir, name).readText()

    val pkg: ModelPackage by lazy { ModelPackage.load { name -> File(assetDir, name).inputStream() } }

    val engine: Engine by lazy { Engine(pkg) }

    // ---- JSON helpers --------------------------------------------------------------------------

    fun doubles(a: JSONArray): DoubleArray = DoubleArray(a.length()) { a.getDouble(it) }

    fun matrix(a: JSONArray): Array<DoubleArray> = Array(a.length()) { doubles(a.getJSONArray(it)) }

    fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

    fun locations(a: JSONArray): List<JSONObject> = (0 until a.length()).map { a.getJSONObject(it) }

    // ---- assertions ----------------------------------------------------------------------------

    fun assertClose(what: String, expected: Double, actual: Double, tol: Double) {
        if (expected.isNaN() || actual.isNaN() || abs(expected - actual) > tol) {
            throw AssertionError("$what: expected $expected but was $actual (tolerance $tol)")
        }
    }

    /** Max abs difference of two `[point][day]` matrices must be <= tol. */
    fun assertMatrixClose(what: String, expected: Array<DoubleArray>, actual: Array<DoubleArray>, tol: Double) {
        if (expected.size != actual.size) throw AssertionError("$what: ${expected.size} rows expected, got ${actual.size}")
        for (k in expected.indices) {
            if (expected[k].size != actual[k].size) {
                throw AssertionError("$what[$k]: ${expected[k].size} values expected, got ${actual[k].size}")
            }
            for (t in expected[k].indices) assertClose("$what[$k][$t]", expected[k][t], actual[k][t], tol)
        }
    }
}
