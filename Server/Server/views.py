import os
from django.conf import settings
from django.http import JsonResponse
from django.http import FileResponse
from django.views.decorators.csrf import csrf_exempt
import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin.messaging import UnregisteredError
from django.apps import apps
import cv2
import json
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from django.apps import apps 
from django.http import HttpResponse

# Use the absolute path to the JSON file 
json_path = os.path.abspath(os.path.join(settings.BASE_DIR, 'push-app-6ba30-firebase-adminsdk-foevy-d8d4ab739f.json'))
cred = credentials.Certificate(json_path)

# Firebase 초기화
if not firebase_admin._apps:  # Firebase 앱이 이미 초기화되었는지 확인
    cred = credentials.Certificate(json_path)
    firebase_admin.initialize_app(cred)

FCM_TOKENS = set([])
a = 0
@csrf_exempt
def upload_image(request):
    global a
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

            # 이미지 처리로직
            input_image = cv2.imread(upload_path)  # 업로드한 이미지를 읽어옴
            predictor = apps.get_app_config('Server')  # 초기화된 AppConfig 인스턴스 가져오기
            # 이미지 URL 생성
            result_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'result.jpg')
            predictor.process_image_to_yolo(input_image, result_path, a)  # 예측 실행

            # print(processed_image)
            # cv2.imwrite(result_path, processed_image)
            if a == 0:
                message = '안전 장비 착용이 확인되었습니다.'
            else:
                message = '안전 장비 착용이 확인되지 않았습니다.'
            a += 1
            print(message)
            # 알림 보내기
            for token in list(FCM_TOKENS):
                send_fcm_message(token, message, message)
            # result_path = r'C:\Users\USER\Documents\GitHub\DSB-Preliminary-Project\Server\media\uploads\1.jpg'    
            return FileResponse(open(result_path, 'rb'), content_type='image/jpeg')

    return JsonResponse({'error': 'Invalid request'}, status=400)

f = True
@csrf_exempt  # CSRF 검사를 비활성화하여 토큰 전송을 받을 수 있도록 함
def add_token(request):
    global f
    if f:
        f = False
        # 이미지 처리로직
        input_image = cv2.imread(r'C:\Users\USER\Documents\GitHub\DSB-Preliminary-Project\Server\media\uploads\1.jpg')  # 업로드한 이미지를 읽어옴
        predictor = apps.get_app_config('Server')  # 초기화된 AppConfig 인스턴스 가져오기
        # 이미지 URL 생성
        result_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'result.jpg')
        predictor.process_image_to_yolo(input_image, result_path, a)  # 예측 실행
    print(len(FCM_TOKENS))
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
    if request.method == 'POST':
        token = request.POST.get('token')
        if token:
            FCM_TOKENS.add(token)
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            return JsonResponse({'message': 'Token received successfully!'}, status=201)
        else:
            return JsonResponse({'error': 'Token not provided'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)

@csrf_exempt
def process_call(request):
    if request.method == 'POST':
        try:
            value = request.POST.getlist('value')  # getlist를 사용하여 리스트 형태로 받음
            # 값이 문자열 형태일 경우 정수로 변환
            value = [int(v) for v in value if v.isdigit()]  # 정수 변환
            result = ''
            index = 0
            if 0 in value:
                result = '화재'
                index+=1
            elif 1 in value:
                if index >= 1:
                    result += ', 추락'
                else:
                    result = '추락'
                    index += 1
            elif 2 in value:
                if index >= 1:
                    result += ', 화재'
                else:
                    result = '화재'
                    index += 1
            elif 3 in value:
                if index >= 1:
                    result += ', 안전장비 미착용'
                else:
                    result = '안전장비 미착용'
            result += ' 상황이 발생하였습니다.'
            print(result)
            # 파일 가져오기
            if 'file' in request.FILES:
                uploaded_file = request.FILES['file']
                file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_file.name)
                # 파일 저장
                with open(file_path, 'wb') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

            # 알림 전송
            for token in list(FCM_TOKENS):
                print(uploaded_file.name, '알림을 전송합니다.')
                send_fcm_message(token, result, result, uploaded_file.name)
            return JsonResponse({'result': result})
        except:
            return JsonResponse({'error': 'Invalid data'}, status=400)
        

def send_fcm_message(token, title, body, image_url=None):
    if image_url != None:
        url = 'http://10.0.2.2:8000/images/' + image_url
        # 메시지 생성
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                image=url
            ),
            token=token,
        )
    else:
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
        # print('알림 전송 성공:', response)
    
    except UnregisteredError:
        print('토큰이 유효하지 않음. 삭제합니다.')
        FCM_TOKENS.remove(token)
        

def serve_image(request, filename):
    # 이미지 파일 경로
    file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
    
    # 파일 존재 여부 확인
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
    else:
        return HttpResponse(status=404)