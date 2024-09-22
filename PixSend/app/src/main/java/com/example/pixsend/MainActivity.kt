package com.example.pixsend

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.bumptech.glide.Glide
import java.io.File
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {
    // Constants에서 서버 URL 불러오기
    private val serverUrl = Constants.SERVER_URL
    // 앱 기능을 위해 필요한 권한 목록
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.POST_NOTIFICATIONS,
    )
    // 권한 요청 구분을 위한 고유 식별 코드
    private val PERMISSION_REQUEST_CODE = 100
    // 카메라 런처
    private lateinit var cameraLauncher: ActivityResultLauncher<Intent>
    // 이미지 선택기 런처
    private lateinit var photoPickerLauncher: ActivityResultLauncher<Intent>
    // 처리된 사진의 Uri를 저장할 변수
    private lateinit var photoURI: Uri
    // 새로고침 경로
    private lateinit var refreshURI: String


    override fun onCreate(savedInstanceState: Bundle?) {
        // 설정 초기화
        super.onCreate(savedInstanceState)
        // 뷰 불러오기
        setContentView(R.layout.activity_main)
        // 네이게이션바와 상단바 제거
        enableEdgeToEdge()
        
        // 휴대폰 UI에 따라 적절한 패딩 추가
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        
        // 앱에서 필요한 권한을 확인 후 요청한다.
        if (!hasAllPermissions()) {
            // 권한이 없는 권한에 대해 요청 수행
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSION_REQUEST_CODE)
        }

        // 기능에 필요한 런처들을 초기화
        initActivityResultLaunchers()
        
        // 카메라 버튼을 찾은 후 기능 처리
        findViewById<Button>(R.id.camera).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                // 권한이 한번 거부되었는지 확인
                if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                    // 권한 요청 다이얼로그를 다시 보여줌
                    ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSION_REQUEST_CODE)
                } else {
                    // 사용자가 한번 거부한 경우 사용자가 직접 수동으로 권한 설정 필요
                    Toast.makeText(this, "카메라 권한이 필요합니다. 설정에서 권한을 허용해주세요.", Toast.LENGTH_LONG).show()
                    // 앱 설정으로 이동
                    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                    val uri = Uri.fromParts("package", packageName, null)
                    intent.data = uri
                    startActivity(intent)
                }
            }
        }

        // 이미지 선택 버튼 클릭 이벤트
        findViewById<Button>(R.id.button_select_image).setOnClickListener {
            // PhotoPicker 오픈
            val intent = Intent(MediaStore.ACTION_PICK_IMAGES)
            photoPickerLauncher.launch(intent)
        }
    }

    /** 권한 체크 함수 */
    private fun hasAllPermissions(): Boolean {
        // 필요한 권한들을 모두 확인하는 반복문
        for (permission in REQUIRED_PERMISSIONS) {
            // 현재 애플리케이션(this)에서 현재 권한(permission)이 부여되어 있는지 확인하며
            // PackageManager.PERMISSION_GRANTED가 아닌 경우는 권한이 부여되지 않는 경우
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false
            }
        }
        return true
    }

    /** 런처 초기화 수행 함수 */
    private fun initActivityResultLaunchers() {
        // 카메라
        cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            // 성공적으로 촬영이 끝난 경우
            if (result.resultCode == RESULT_OK) {
                refreshGallery(refreshURI)
            }
        }
        
        // 이미지 선택기
        photoPickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                // 선택된 이미지 URI 처리
                val selectedImageUri: Uri? = result.data?.data
                // 선택된 이미지 처리 로직 설정
                selectedImageUri?.let { uri ->
                    uploadImageToServer(uri)
                }
            }
        }
    }

    /**
     * 선택한 이미지를 서버에 업로드하여 처리 결과를 받아온다
     * @param imageUri 업로드할 이미지의 Uri
     */
    private fun uploadImageToServer(imageUri: Uri) {
        // 선택된 이미지의 실제 경로 가져오기
        val filePath = getRealPathFromURI(imageUri) // Uri에서 파일 경로를 가져옵니다.
        val file = File(filePath) // 파일 객체를 생성합니다.

        // OkHttp 클라이언트 설정
        val client = OkHttpClient() // OkHttp 클라이언트를 생성합니다.
        val mediaType = "image/jpeg".toMediaTypeOrNull() // 미디어 타입을 설정합니다.

        // 멀티파트 요청 본체를 생성합니다.
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM) // 폼 데이터로 설정합니다.
            .addFormDataPart("image", file.name, RequestBody.create(mediaType, file)) // 파일을 요청 본체에 추가합니다.
            .build()

        // 서버에 요청을 구성합니다.
        val request = Request.Builder()
            .url("$serverUrl/upload/") // 서버 URL을 설정합니다.
            .post(requestBody) // POST 요청으로 설정합니다.
            .build()

        // 서버로 이미지 업로드 요청 보내기
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // 요청 실패 시 로그에 오류 메시지를 기록합니다.
                Log.e("Upload Error", e.message ?: "Error")
            }

            override fun onResponse(call: Call, response: Response) {
                // 서버 응답이 성공적일 경우 처리합니다.
                if (response.isSuccessful) {
                    handleUploadResponse(response) // 응답 처리 함수 호출
                } else {
                    // 응답이 실패할 경우 로그에 오류 메시지를 기록합니다.
                    Log.e("Upload Error", "Failed to upload image: ${response.code}")
                }
            }
        })
    }

    /**
     * Uri에서 실제 파일 경로를 가져오는 함수
     * 주어진 Uri를 사용하여 해당 이미지 파일의 실제 경로를 반환합니다.
     */
    private fun getRealPathFromURI(uri: Uri): String? {
        var realPath: String? = null
        // contentResolver를 사용하여 Uri에 대한 쿼리를 수행합니다.
        val cursor = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            // 쿼리 결과에서 첫 번째 행으로 이동합니다.
            if (it.moveToFirst()) {
                // DATA 열의 인덱스를 가져옵니다.
                val idx = it.getColumnIndex(MediaStore.Images.Media.DATA)
                // 인덱스가 유효하면 해당 경로를 가져옵니다.
                realPath = if (idx != -1) it.getString(idx) else null
            }
        }
        // 실제 경로를 반환합니다. 경로가 없으면 null을 반환합니다.
        return realPath
    }

    /**
     * 서버에서 이미지 업로드 응답을 처리하는 함수
     * 서버의 응답을 통해 이미지를 다운로드하고 갤러리에 저장 후 화면에 표시합니다.
     */
    private fun handleUploadResponse(response: Response) {
        val baseName = "result_image" // 기본 파일 이름
        val extension = ".jpg" // 파일 확장자
        val uniqueFileName = getUniqueFileName(baseName, extension) // 중복되지 않는 파일 이름 생성

        // 지정된 디렉토리에 고유한 파일 이름으로 파일 객체 생성
        val file = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "PixSend/$uniqueFileName")

        // 응답의 바디에서 바이트 스트림을 가져와서 파일로 저장합니다.
        response.body?.byteStream()?.use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                // 입력 스트림에서 출력 스트림으로 데이터를 복사합니다.
                inputStream.copyTo(outputStream)
            }
        }

        // 갤러리를 새로 고칩니다. 저장된 파일의 경로를 사용합니다.
        refreshGallery(file.absolutePath)

        // UI 스레드에서 이미지 뷰에 이미지를 로드합니다.
        runOnUiThread {
            // Glide를 사용하여 이미지를 로드하고 ImageView에 표시합니다.
            Glide.with(this).load(file).into(findViewById<ImageView>(R.id.imageView))
        }
    }

    /** 사진을 찍은 후 갤러리를 새로고침 하는 함수 */
    private fun refreshGallery(filePath: String) {
        MediaScannerConnection.scanFile(this, arrayOf(filePath), null) { path, uri ->
            Log.d("MediaScan", "Scan $path:\n-> uri=$uri")
        }
    }

    /** 카메라 오픈 기능 처리 */
    private fun openCamera() {
        // 사진이 저장될 경로 지정
        val directory = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "PixSend")
        // 디렉토리가 없으면 생성
        if (!directory.exists()) {
            directory.mkdirs()
        }
        // 중복 x 파일이름을 생성
        val uniqueFileName = getUniqueFileName("PixSend_Image", ".jpg")
        Log.d("uniqueFileName", uniqueFileName)

        // 생성할 파일의 경로를 생성
        val file = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "PixSend/$uniqueFileName")
        // 지정된 파일에 대한 콘텐츠 URI를 반환
        photoURI = FileProvider.getUriForFile(this, "${packageName}.provider", file)
        refreshURI = file.absolutePath

        // Intent를 통해서 다른 컴포넌트(카메라)를 호출(MediaStore.ACTION_IMAGE_CAPTURE)한다.
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            // 카메라 앱에서 photoURI에 사진을 저장하도록 지정
            putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
        }
        // 카메라 런처를 통해 작업 수행
        cameraLauncher.launch(cameraIntent)
    }

    /** 이미지 개수 파악 후 이름이 중복되지 않는 이름을 반환한다. */
    fun getUniqueFileName(baseName: String, extension: String): String {
        val directory = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "PixSend")
        if (!directory.exists()) {
            directory.mkdirs() // 디렉토리가 없으면 생성
        }

        // 기본 파일 이름과 확장자를 결합
        var uniqueFileName = "$baseName$extension"
        var count = 1

        // 같은 이름의 파일이 존재하는지 확인
        while (File(directory, uniqueFileName).exists()) {
            uniqueFileName = "$baseName($count)$extension" // 예: captured_image(1).jpg
            count++
        }

        return uniqueFileName
    }
}