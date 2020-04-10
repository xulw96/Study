from django.contrib import admin
from django.urls import path, include

from AXF import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('axf/', include(('App.urls', 'App'), namespace='axf'))
]

if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        path('__debug__/', include(debug_toolbar.urls))
    ] + urlpatterns