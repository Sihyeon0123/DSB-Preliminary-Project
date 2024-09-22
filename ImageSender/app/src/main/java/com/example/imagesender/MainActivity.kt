package com.example.imagesender

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.bumptech.glide.Glide
import com.google.firebase.messaging.FirebaseMessaging
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private val serverUrl = "http://10.0.2.2:8000/"
    private lateinit var selectedImageUri: Uri
    private lateinit var photoURI: Uri

    private lateinit var galleryLauncher: ActivityResultLauncher<Intent>
    private lateinit var cameraLauncher: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initActivityResultLaunchers()
        initFirebaseMessaging()

        findViewById<Button>(R.id.camera).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
            }
        }

        findViewById<Button>(R.id.button_select_image).setOnClickListener {
            if (checkStoragePermissions()) {
                openGallery()
            } else {
                requestStoragePermissions()
            }
        }
    }

    private fun initActivityResultLaunchers() {
        galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                result.data?.data?.let { uri -> uploadImageToServer(uri) }
            }
        }

        cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                refreshGallery(photoURI.toString())
            }
        }
    }

    private fun initFirebaseMessaging() {
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w(TAG, "Fetching FCM registration token failed", task.exception)
                return@addOnCompleteListener
            }
            val token = task.result
            sendTokenToServer(token)
            Toast.makeText(baseContext, "FCM 토큰: $token", Toast.LENGTH_SHORT).show()
        }
    }

    private fun openCamera() {
        val file = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "captured_image.jpg")
        photoURI = FileProvider.getUriForFile(this, "${packageName}.provider", file)

        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
        }
        cameraLauncher.launch(cameraIntent)
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK).apply {
            setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*")
        }
        galleryLauncher.launch(intent)
    }

    private fun uploadImageToServer(imageUri: Uri) {
        val filePath = getRealPathFromURI(imageUri)
        val file = File(filePath)

        val client = OkHttpClient()
        val mediaType = "image/jpeg".toMediaTypeOrNull()
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", file.name, RequestBody.create(mediaType, file))
            .build()

        val request = Request.Builder()
            .url("$serverUrl/upload/")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("Upload Error", e.message ?: "Error")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    handleUploadResponse(response)
                } else {
                    Log.e("Upload Error", "Failed to upload image: ${response.code}")
                }
            }
        })
    }

    private fun handleUploadResponse(response: Response) {
        val file = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "downloaded_image.jpg")
        response.body?.byteStream()?.use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        refreshGallery(file.absolutePath)
        runOnUiThread {
            Glide.with(this).load(file).into(findViewById<ImageView>(R.id.imageView))
        }
    }

    private fun getRealPathFromURI(contentUri: Uri): String {
        val proj = arrayOf(MediaStore.Images.Media.DATA)
        contentResolver.query(contentUri, proj, null, null, null)?.use { cursor ->
            cursor.moveToFirst()
            val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
            return cursor.getString(columnIndex)
        }
        return ""
    }

    private fun refreshGallery(filePath: String) {
        MediaScannerConnection.scanFile(this, arrayOf(filePath), null) { path, uri ->
            Log.d("MediaScanner", "Scanned $path:\n-> uri=$uri")
        }
    }

    private fun sendTokenToServer(token: String) {
        val client = OkHttpClient()
        val requestBody = FormBody.Builder().add("token", token).build()

        val request = Request.Builder()
            .url("$serverUrl/addToken/")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("Token Upload Error", e.message ?: "Error")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    Log.d("Token Upload", "Token sent successfully: ${response.body?.string()}")
                } else {
                    Log.e("Token Upload Error", "Failed to send token: ${response.code}")
                }
            }
        })
    }

    private fun checkStoragePermissions(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestStoragePermissions() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE), STORAGE_PERMISSION_CODE)
    }

    companion object {
        private const val STORAGE_PERMISSION_CODE = 100
        private const val CAMERA_PERMISSION_CODE = 100
        private const val TAG = "MainActivity"
    }
}