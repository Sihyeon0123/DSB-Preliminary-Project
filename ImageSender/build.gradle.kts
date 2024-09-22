// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    // 알림 기능: 구글 서비스
    id("com.google.gms.google-services") version "4.4.2" apply false
}