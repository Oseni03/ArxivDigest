from django.urls import path
from django.views.generic import TemplateView


app_name = "arxiv"

urlspatterns = [
    path('', TemplateView.as_view(template_name='arxiv/home.html')),
]