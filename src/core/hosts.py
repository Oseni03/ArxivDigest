from django_hosts import patterns, host
from django.conf import settings

host_patterns = patterns(
    '',
    host(r"admin", "core.urls.admin", name="admin"),
    host(r'arxiv', 'arxiv.urls', name='arxiv'),
    host(r'books', 'books.urls', name='books'),
    host(r"", settings.ROOT_URLCONF, name="main"),
)