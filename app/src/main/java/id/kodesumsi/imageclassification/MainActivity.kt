package id.kodesumsi.imageclassification

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.floatingactionbutton.FloatingActionButton
import id.kodesumsi.imageclassification.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min


class MainActivity : AppCompatActivity() {

    private lateinit var label: TextView
    private lateinit var takePhoto: FloatingActionButton
    private lateinit var preview: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // init component
        label = findViewById(R.id.label)
        takePhoto = findViewById(R.id.btn_photo)
        preview = findViewById(R.id.preview)

        // register activity register for result
        val onCameraResult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                var image: Bitmap = result.data?.extras?.get("data") as Bitmap
                val dimension = min(image.width, image.height)
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                preview.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false)
                classifyImage(image)
            }
        }

        // take photo action
        takePhoto.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                onCameraResult.launch(cameraIntent)
            } else {
                // Request camera permission
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

    }

    private fun classifyImage(image: Bitmap?) {
        val model = Model.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // get 1D array of 224 * 224 pixels in image
        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        image!!.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)

        // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
        var pixel = 0
        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
            }
        }

        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val confidences = outputFeature0.floatArray

        // find the index of the class with the biggest confidence.
        var maxPos = 0
        var maxConfidence = 0f
        for (i in confidences.indices) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxPos = i
            }
        }

        // label kelasnya apa aja
        val classes = arrayOf("Banana", "Orange", "Pen", "Sticky Notes")
        label.text = classes[maxPos]

        // Releases model resources if no longer used.
        model.close()
    }

    companion object {
        const val IMAGE_SIZE = 224;
    }

}