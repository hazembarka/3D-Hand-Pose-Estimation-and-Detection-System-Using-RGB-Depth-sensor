from django.http import HttpResponse
from django.shortcuts import render
from .process import begin
def home(request):
    begin()
    return HttpResponse('Hello, World!')

def template(request):
    return render(request,'hello_world.html')