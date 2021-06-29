from django.urls import path
from .import views
app_name = 'post'
urlpatterns = [
    path('', views.filter, name='filter'),
    path('obj-detect/', views.objDt, name='objDt'),
    path('obj-counting/', views.objCounting, name='counting'),
    path('face-detect/', views.faceDt, name='faceDetect'),
]
