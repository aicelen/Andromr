from kivy.utils import platform
import os


def save_to_external_storage(path, dest_filename=None):
    """
    Save an existing XML file to Android external storage using MediaStore API.

    Args:
        path (str): Path to the existing XML file
        dest_filename (str, optional): The destination filename to use.
                                       If None, uses the original filename.

    Returns:
        str: Path to the saved file or error message
    """

    if platform != "android":
        raise RuntimeError("Saving a file to mediastore only works on Android")

    try:
        # Check if source file exists
        if not os.path.exists(path):
            return f"Error: Source file '{path}' not found."

        # Import Android-specific modules
        from jnius import autoclass  # pylint: disable=import-error # type: ignore

        # Get filename from path if not provided
        if not dest_filename:
            dest_filename = os.path.basename(path)

        # Get Android context
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        context = PythonActivity.mActivity.getApplicationContext()

        # Import required Android classes
        Environment = autoclass("android.os.Environment")
        MediaFiles = autoclass("android.provider.MediaStore$Files")
        MediaColumns = autoclass("android.provider.MediaStore$MediaColumns")
        ContentValues = autoclass("android.content.ContentValues")
        File = autoclass("java.io.File")
        FileInputStream = autoclass("java.io.FileInputStream")
        FileOutputStream = autoclass("java.io.FileOutputStream")
        BufferedInputStream = autoclass("java.io.BufferedInputStream")
        BufferedOutputStream = autoclass("java.io.BufferedOutputStream")

        # Read source file
        source_file = File(path)

        # Check if Android 10 (API 29) or higher
        Build = autoclass("android.os.Build$VERSION")
        if Build.SDK_INT >= 29:  # Android 10+
            # Set up ContentValues for MediaStore
            content_values = ContentValues()
            content_values.put(MediaColumns.DISPLAY_NAME, dest_filename)
            content_values.put(MediaColumns.MIME_TYPE, "application/octet-stream")
            content_values.put(
                MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS
            )

            # Insert the file into MediaStore
            resolver = context.getContentResolver()
            uri = resolver.insert(MediaFiles.getContentUri("external"), content_values)

            if uri:
                # Copy file content using streams
                input_stream = BufferedInputStream(FileInputStream(source_file))
                output_stream = BufferedOutputStream(resolver.openOutputStream(uri))

                # Create buffer for copying
                buffer_size = 8192
                buffer = bytearray(buffer_size)
                bytes_read = input_stream.read(buffer, 0, buffer_size)

                # Copy in chunks
                while bytes_read != -1:
                    output_stream.write(buffer, 0, bytes_read)
                    bytes_read = input_stream.read(buffer, 0, buffer_size)

                # Close streams
                output_stream.flush()
                output_stream.close()
                input_stream.close()

                # Get the actual file path (for reference only)
                cursor = resolver.query(uri, [MediaColumns.DATA], None, None, None)
                if cursor:
                    cursor.close()

                return "XML file saved successfully in Documents"
            else:
                return "Failed to create file in MediaStore."

        else:  # Android 9 or lower - using traditional file access
            # Get Documents directory path
            docs_dir = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOCUMENTS
            ).getAbsolutePath()

            # Create destination path
            dest_path = os.path.join(docs_dir, dest_filename)
            dest_file = File(dest_path)

            # Create parent directories if they don't exist
            parent_dir = dest_file.getParentFile()
            if not parent_dir.exists():
                parent_dir.mkdirs()

            # Copy file content
            input_stream = BufferedInputStream(FileInputStream(source_file))
            output_stream = BufferedOutputStream(FileOutputStream(dest_file))

            buffer_size = 8192
            buffer = bytearray(buffer_size)
            bytes_read = input_stream.read(buffer, 0, buffer_size)

            while bytes_read != -1:
                output_stream.write(buffer, 0, bytes_read)
                bytes_read = input_stream.read(buffer, 0, buffer_size)

            # Close streams
            output_stream.flush()
            output_stream.close()
            input_stream.close()

            # Make the file visible in MediaStore
            values = ContentValues()
            values.put(MediaColumns.DATA, dest_path)
            values.put(MediaColumns.MIME_TYPE, "application/octet-stream")
            values.put(MediaColumns.DISPLAY_NAME, dest_filename)

            resolver = context.getContentResolver()
            resolver.insert(MediaFiles.getContentUri("external"), values)

            return f"XML file saved successfully in Documents"

    except Exception as e:
        return f"Error saving XML file: {e}"
