import android.content.ContentUris
import android.provider.MediaStore
import com.example.lamforgallery.database.AppDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// Call this function once after copying the DB to the new device
suspend fun resyncDatabaseOnNewDevice(context: android.content.Context) {
    val db = AppDatabase.getDatabase(context)
    val contentResolver = context.contentResolver

    withContext(Dispatchers.IO) {
        // 1. Scan the NEW device to find all actual images and their metadata
        // Map Key: "DateTaken_Width_Height" -> Value: New Device URI
        val deviceImageMap = mutableMapOf<String, String>()

        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_TAKEN,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT
        )

        contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            null
        )?.use { cursor ->
            val idCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val dateCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN)
            val wCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.WIDTH)
            val hCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.HEIGHT)

            while (cursor.moveToNext()) {
                val id = cursor.getLong(idCol)
                val date = cursor.getLong(dateCol)
                val w = cursor.getInt(wCol)
                val h = cursor.getInt(hCol)
                val newUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id).toString()
                
                // Create a unique signature for the image
                val signature = "${date}_${w}_${h}"
                deviceImageMap[signature] = newUri
            }
        }

        // 2. Fetch all existing DB entries (which currently point to the OLD device's URIs)
        val allEmbeddings = db.imageEmbeddingDao().getAllEmbeddings()
        val allPeople = db.personDao().getAllPeople()

        // 3. Iterate and Update
        // We use a raw transaction or direct DAO calls to swap the URIs
        
        allEmbeddings.forEach { embedding ->
    // Create signature from the DB data
    val signature = "${embedding.dateTaken}_${embedding.width}_${embedding.height}"
    val newUri = deviceImageMap[signature]

    if (newUri != null && newUri != embedding.uri) {
        db.runInTransaction {
            // 1. Update the main embedding table
            // Note: We might need to handle constraints if 'newUri' already exists (e.g. duplicates), 
            // but for a clean transfer, it shouldn't.
            try {
                db.imageEmbeddingDao().updateUri(embedding.uri, newUri)
                
                // 2. Update the linking table (Many-to-Many)
                db.personDao().updateCrossRefUri(embedding.uri, newUri)
                
                // 3. Update any Person covers
                db.personDao().updateCoverUri(embedding.uri, newUri)
                
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
}
    }
}
