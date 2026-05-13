from jnius import (  # pylint: disable=import-error # type: ignore
    PythonJavaClass,
    autoclass,
    cast,
    java_method,
)

from androidstorage4kivy import SharedStorage  # type: ignore
from homr.simple_logging import eprint

REQUEST_CODE = 43
RESULT_OK = -1

PythonActivity = autoclass("org.kivy.android.PythonActivity")
GmsDocumentScannerOptions = autoclass(
    "com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions"
)
GmsDocumentScannerOptionsBuilder = autoclass(
    "com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions$Builder"
)
GmsDocumentScanning = autoclass("com.google.mlkit.vision.documentscanner.GmsDocumentScanning")
GmsDocumentScanningResult = autoclass(
    "com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult"
)


class ScanIntentSuccessListener(PythonJavaClass):
    __javainterfaces__ = ("com.google.android.gms.tasks.OnSuccessListener",)
    __javacontext__ = "app"

    def __init__(self, on_error):
        super().__init__()
        self.on_error = on_error

    @java_method("(Ljava/lang/Object;)V")
    def onSuccess(self, intent_sender):
        activity = PythonActivity.mActivity
        activity.startIntentSenderForResult(
            cast("android.content.IntentSender", intent_sender),
            REQUEST_CODE,
            None,
            0,
            0,
            0,
        )


class ScanIntentFailureListener(PythonJavaClass):
    __javainterfaces__ = ("com.google.android.gms.tasks.OnFailureListener",)
    __javacontext__ = "app"

    def __init__(self, on_error):
        super().__init__()
        self.on_error = on_error

    @java_method("(Ljava/lang/Exception;)V")
    def onFailure(self, exception):
        eprint("Scanning failed")


def start_document_scan(on_error=None):
    """
    Launch Google Play services' ML Kit document scanner.

    The result is delivered through Android's activity result callback and can
    be consumed with handle_document_scan_result().
    """
    options = (
        GmsDocumentScannerOptionsBuilder()
        .setGalleryImportAllowed(False)
        .setResultFormats(GmsDocumentScannerOptions.RESULT_FORMAT_JPEG)
        .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
        .build()
    )
    scanner = GmsDocumentScanning.getClient(options)
    success_listener = ScanIntentSuccessListener(on_error)
    failure_listener = ScanIntentFailureListener(on_error)
    scanner.getStartScanIntent(PythonActivity.mActivity).addOnSuccessListener(
        success_listener
    ).addOnFailureListener(failure_listener)


def document_scan_result(data):
    intent = cast("android.content.Intent", data)
    result = GmsDocumentScanningResult.fromActivityResultIntent(intent)
    pages = result.getPages()

    paths = []
    shared_storage_class = SharedStorage()
    for index in range(pages.size()):
        page = pages.get(index)
        uri = page.getImageUri()
        paths.append(shared_storage_class._copy_uri_to_cache(uri))
    return paths
