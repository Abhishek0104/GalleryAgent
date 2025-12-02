package com.example.lamforgallery.ui

import android.Manifest
import android.app.Activity.RESULT_OK
import android.content.ContentUris // --- NEW IMPORT
import android.content.Context     // --- NEW IMPORT
import android.content.IntentSender
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore // --- NEW IMPORT
import android.util.Log            // --- NEW IMPORT
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChatBubble
import androidx.compose.material.icons.filled.Face 
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.PhotoAlbum
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.lamforgallery.database.AppDatabase // --- NEW IMPORT
import kotlinx.coroutines.Dispatchers               // --- NEW IMPORT
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext              // --- NEW IMPORT
import java.net.URLDecoder
import java.nio.charset.StandardCharsets

class MainActivity : ComponentActivity() {

    private val factory by lazy { ViewModelFactory(application) }

    private val agentViewModel: AgentViewModel by viewModels { factory }
    private val photosViewModel: PhotosViewModel by viewModels { factory }
    private val albumsViewModel: AlbumsViewModel by viewModels { factory }
    private val embeddingViewModel: EmbeddingViewModel by viewModels { factory }
    private val photoViewerViewModel: PhotoViewerViewModel by viewModels { factory }
    private val peopleViewModel: PeopleViewModel by viewModels { factory }

    private val permissionToRequest = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        Manifest.permission.READ_MEDIA_IMAGES
    } else {
        Manifest.permission.READ_EXTERNAL_STORAGE
    }
    private var isPermissionGranted by mutableStateOf(false)

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        isPermissionGranted = isGranted
        if (isGranted) loadAllViewModels()
    }

    private var currentPermissionType: PermissionType? = null
    private val permissionRequestLauncher = registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { activityResult ->
        val type = currentPermissionType
        if (type != null) {
            val wasSuccessful = activityResult.resultCode == RESULT_OK
            agentViewModel.onPermissionResult(wasSuccessful, type)
            currentPermissionType = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // --- DATABASE MIGRATION TRIGGER ---
        // Set this to TRUE only when running on the target device after copying the DB files.
        // It will fix the file paths to match the new device.
        val isDemoTransferMode = false 
        
        if (isDemoTransferMode) {
             lifecycleScope.launch {
                 resyncDatabaseOnNewDevice(applicationContext)
             }
        }
        // -----------------------------------

        checkAndRequestPermission()
        lifecycleScope.launch {
            agentViewModel.galleryDidChange.collect {
                photosViewModel.loadPhotos()
                albumsViewModel.loadAlbums()
            }
        }
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    if (isPermissionGranted) {
                        AppNavigationHost(
                            factory = factory,
                            agentViewModel = agentViewModel,
                            photosViewModel = photosViewModel,
                            albumsViewModel = albumsViewModel,
                            embeddingViewModel = embeddingViewModel,
                            photoViewerViewModel = photoViewerViewModel,
                            peopleViewModel = peopleViewModel, 
                            onLaunchPermissionRequest = { intentSender, type ->
                                currentPermissionType = type
                                permissionRequestLauncher.launch(IntentSenderRequest.Builder(intentSender).build())
                            }
                        )
                    } else {
                        PermissionDeniedScreen { requestPermissionLauncher.launch(permissionToRequest) }
                    }
                }
            }
        }
    }

    private fun checkAndRequestPermission() {
        if (ContextCompat.checkSelfPermission(this, permissionToRequest) == PackageManager.PERMISSION_GRANTED) {
            isPermissionGranted = true
            loadAllViewModels()
        } else {
            requestPermissionLauncher.launch(permissionToRequest)
        }
    }

    private fun loadAllViewModels() {
        photosViewModel.loadPhotos()
        albumsViewModel.loadAlbums()
    }
}

/**
 * Scans the current device for images and updates the database to point to the new URIs.
 * This matches images based on Date Taken + Width + Height.
 */
suspend fun resyncDatabaseOnNewDevice(context: Context) {
    Log.d("DB_SYNC", "Starting Database Resync...")
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

        try {
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
                    // We use a simplified date (divide by 1000) to allow for minor second-level differences if any
                    // But usually exact timestamp matches.
                    val signature = "${date}_${w}_${h}"
                    deviceImageMap[signature] = newUri
                }
            }
        } catch (e: Exception) {
            Log.e("DB_SYNC", "Failed to scan media", e)
            return@withContext
        }
        
        Log.d("DB_SYNC", "Found ${deviceImageMap.size} images on this device.")

        // 2. Fetch all existing DB entries (which currently point to the OLD device's URIs)
        val allEmbeddings = db.imageEmbeddingDao().getAllEmbeddings()
        var updatedCount = 0

        // 3. Iterate and Update
        db.runInTransaction {
            allEmbeddings.forEach { embedding ->
                // Construct the signature from the DB data
                val signature = "${embedding.dateTaken}_${embedding.width}_${embedding.height}"
                val newUri = deviceImageMap[signature]

                if (newUri != null && newUri != embedding.uri) {
                    try {
                        // A. Update the Main Embedding Table
                        // (We assume the new URI doesn't exist in the DB yet, because we copied the DB from another device)
                        db.imageEmbeddingDao().updateUri(embedding.uri, newUri)

                        // B. Update the Cross-Reference Table (Image <-> Person)
                        db.personDao().updateCrossRefUri(embedding.uri, newUri)

                        // C. Update Person Cover Photos
                        db.personDao().updateCoverUri(embedding.uri, newUri)
                        
                        updatedCount++
                    } catch (e: Exception) {
                        Log.e("DB_SYNC", "Failed to update ${embedding.uri}", e)
                    }
                }
            }
        }
        Log.d("DB_SYNC", "Resync Complete. Updated $updatedCount paths.")
    }
}

@Composable
fun AppNavigationHost(
    factory: ViewModelProvider.Factory,
    agentViewModel: AgentViewModel,
    photosViewModel: PhotosViewModel,
    albumsViewModel: AlbumsViewModel,
    embeddingViewModel: EmbeddingViewModel,
    photoViewerViewModel: PhotoViewerViewModel,
    peopleViewModel: PeopleViewModel, 
    onLaunchPermissionRequest: (IntentSender, PermissionType) -> Unit
) {
    val navController = rememberNavController()
    var selectedTab by rememberSaveable { mutableStateOf("photos") }

    NavHost(navController = navController, startDestination = "main") {
        composable("main") {
            AppShell(
                agentViewModel = agentViewModel,
                photosViewModel = photosViewModel,
                albumsViewModel = albumsViewModel,
                embeddingViewModel = embeddingViewModel,
                peopleViewModel = peopleViewModel, 
                selectedTab = selectedTab,
                onTabSelected = { selectedTab = it },
                onAlbumClick = { encodedName -> navController.navigate("album_detail/$encodedName") },
                onLaunchPermissionRequest = onLaunchPermissionRequest,
                onPhotoClick = { encodedUri ->
                    val currentList = photosViewModel.uiState.value.photos
                    val uri = try {
                        URLDecoder.decode(encodedUri, StandardCharsets.UTF_8.name())
                    } catch(e:Exception) { encodedUri }

                    photoViewerViewModel.setPhotoList(currentList, uri)
                    navController.navigate("view_photo")
                },
                onNavigateToCleanup = { navController.navigate("cleanup_review") }
            )
        }

        composable(
            route = "album_detail/{albumName}",
            arguments = listOf(navArgument("albumName") { type = NavType.StringType })
        ) { backStackEntry ->
            val albumName = backStackEntry.arguments?.getString("albumName") ?: "Unknown"
            val albumDetailViewModel: AlbumDetailViewModel = androidx.lifecycle.viewmodel.compose.viewModel(factory = factory)
            AlbumDetailScreen(
                albumName = albumName,
                viewModel = albumDetailViewModel,
                onNavigateBack = { navController.popBackStack() },
                onPhotoClick = { uri ->
                    val currentList = albumDetailViewModel.uiState.value.photos
                    photoViewerViewModel.setPhotoList(currentList, uri)
                    navController.navigate("view_photo")
                }
            )
        }

        composable("view_photo") {
            SinglePhotoScreen(
                viewModel = photoViewerViewModel,
                onNavigateBack = { navController.popBackStack() },
                onAgentAction = { prompt, uri ->
                    agentViewModel.setExternalSelection(listOf(uri))
                    navController.popBackStack("main", inclusive = false)
                    selectedTab = "agent"
                    agentViewModel.sendUserInput(prompt)
                }
            )
        }

        composable("cleanup_review") {
            val uiState by agentViewModel.uiState.collectAsState()
            CleanupScreen(
                duplicateGroups = uiState.cleanupGroups,
                onNavigateBack = { navController.popBackStack() },
                onConfirmDelete = { urisToDelete ->
                    navController.popBackStack()
                    agentViewModel.setExternalSelection(urisToDelete)
                    agentViewModel.sendUserInput("Delete these duplicates")
                }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppShell(
    agentViewModel: AgentViewModel,
    photosViewModel: PhotosViewModel,
    albumsViewModel: AlbumsViewModel,
    embeddingViewModel: EmbeddingViewModel,
    peopleViewModel: PeopleViewModel, 
    selectedTab: String,
    onTabSelected: (String) -> Unit,
    onAlbumClick: (String) -> Unit,
    onPhotoClick: (String) -> Unit,
    onLaunchPermissionRequest: (IntentSender, PermissionType) -> Unit,
    onNavigateToCleanup: () -> Unit
) {
    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f),
            ) {
                NavigationBarItem(
                    selected = selectedTab == "photos",
                    onClick = { onTabSelected("photos") },
                    icon = { Icon(Icons.Default.Photo, "Photos") },
                    label = { Text("Photos") }
                )
                NavigationBarItem(
                    selected = selectedTab == "albums",
                    onClick = { onTabSelected("albums") },
                    icon = { Icon(Icons.Default.PhotoAlbum, "Albums") },
                    label = { Text("Albums") }
                )
                
                NavigationBarItem(
                    selected = selectedTab == "people",
                    onClick = { onTabSelected("people") },
                    icon = { Icon(Icons.Default.Face, "People") },
                    label = { Text("People") }
                )
                NavigationBarItem(
                    selected = selectedTab == "agent",
                    onClick = { onTabSelected("agent") },
                    icon = { Icon(Icons.Default.ChatBubble, "Agent") },
                    label = { Text("Agent") }
                )
                NavigationBarItem(
                    selected = selectedTab == "indexing",
                    onClick = { onTabSelected("indexing") },
                    icon = { Icon(Icons.Default.ImageSearch, "Indexing") },
                    label = { Text("Indexing") }
                )
            }
        }
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            when (selectedTab) {
                "photos" -> PhotosScreen(
                    viewModel = photosViewModel,
                    onSendToAgent = { uris -> agentViewModel.setExternalSelection(uris); onTabSelected("agent") },
                    onPhotoClick = onPhotoClick
                )
                "albums" -> AlbumsScreen(viewModel = albumsViewModel, onAlbumClick = onAlbumClick)
                "people" -> PeopleScreen(viewModel = peopleViewModel) 
                "agent" -> AgentScreen(
                    viewModel = agentViewModel,
                    onLaunchPermissionRequest = onLaunchPermissionRequest,
                    onNavigateToCleanup = onNavigateToCleanup
                )
                "indexing" -> EmbeddingScreen(viewModel = embeddingViewModel)
            }
        }
    }
}

@Composable
fun PermissionDeniedScreen(onRequestPermission: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Permission Required")
            Spacer(Modifier.height(8.dp))
            Button(onClick = onRequestPermission) {
                Text("Grant Access")
            }
        }
    }
}
