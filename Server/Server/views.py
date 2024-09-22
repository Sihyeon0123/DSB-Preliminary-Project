import os
from django.conf import settings
from django.http import JsonResponse
from django.http import FileResponse
from django.views.decorators.csrf import csrf_exempt
import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin.messaging import UnregisteredError

# Use the absolute path to the JSON file 
json_path = os.path.abspath(os.path.join(settings.BASE_DIR, 'push-app-6ba30-firebase-adminsdk-foevy-d8d4ab739f.json'))
cred = credentials.Certificate(json_path)

# Firebase 초기화
if not firebase_admin._apps:  # Firebase 앱이 이미 초기화되었는지 확인
    cred = credentials.Certificate(json_path)
    firebase_admin.initialize_app(cred)

FCM_TOKENS = set([])

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            # 업로드 경로 설정
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)  # 폴더가 없으면 생성

            # 파일명 고정이 아닌 유니크하게 저장 (중복 방지)
            upload_path = os.path.join(upload_dir, f"uploaded_{image_file.name}")
            with open(upload_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # 이미지 처리로직 추가...

            # 이미지 URL 생성
            result_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'result.jpg')

            # 알림 보내기
            for token in list(FCM_TOKENS):
                send_fcm_message(token, '이미지 업로드 성공', '새 이미지가 업로드되었습니다.')

            return FileResponse(open(result_path, 'rb'), content_type='image/jpeg')

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt  # CSRF 검사를 비활성화하여 토큰 전송을 받을 수 있도록 함
def add_token(request):
    print(len(FCM_TOKENS))
    if request.method == 'POST':
        token = request.POST.get('token')
        if token:
            FCM_TOKENS.add(token)

            return JsonResponse({'message': 'Token received successfully!'}, status=201)
        else:
            return JsonResponse({'error': 'Token not provided'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)


def send_fcm_message(token, title, body):
    # 메시지 생성
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        token=token,
    )
    
    try:
        # 메시지 전송
        response = messaging.send(message)
        print('알림 전송 성공:', response)
    
    except UnregisteredError:
        print('토큰이 유효하지 않음. 삭제합니다.')
        FCM_TOKENS.remove(token)
        