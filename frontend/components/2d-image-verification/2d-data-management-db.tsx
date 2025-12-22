"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useImageAnnotations } from "@/hooks/api"
import { validateName, getNameValidationMessage, isNameDuplicate, getDuplicateNameMessage } from "@/lib/validation"
import {
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import {
  Database,
  Upload,
  RefreshCw,
  Search,
  LayoutGrid,
  List,
  Folder,
  FolderOpen,
  FileImage,
  FileText,
  Image as ImageIcon,
  Info,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Eye,
} from "lucide-react"
import { AnnotatedImageViewer } from "@/components/annotations/AnnotatedImageViewer"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"

interface Dataset {
  id: string
  name: string
  description?: string
  type: string
  source?: string
  size: number
  storageLocation?: string
  metadata?: {
    model?: string
  }
  createdAt: string
  updatedAt: string
  imageFiles?: ImageFile[]
}

interface ImageFile {
  id: string
  datasetId: string
  filename: string
  data?: string // Base64 encoded image data
  storage_key?: string // Storage path for direct API access
  previewUrl?: string | null
  width?: number
  height?: number
  format: string
  mimeType: string
  metadata?: any
  createdAt: string
  detections?: ImageDetection[]
  visualization?: string
}

interface ImageDetection {
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
  class: string
  class_id: number
  confidence: number
  isGroundTruth?: boolean
}

interface ModelImageMetadata {
  filename: string
  visualization?: string
  detections: ImageDetection[]
}

interface ModelMetadataResponse {
  model: string
  timestamp: string
  target_class: string
  class_id: number
  images: ModelImageMetadata[]
}

type FileTreeNode = {
  name: string
  type: "folder" | "file"
  children?: Record<string, FileTreeNode>
  file?: ImageFile
}

type DisplayItem =
  | { type: "dataset"; dataset: Dataset }
  | { type: "file"; file: ImageFile; metadata?: ModelImageMetadata }

const buildFileTree = (files: ImageFile[]): FileTreeNode => {
  const root: FileTreeNode = { name: "root", type: "folder", children: {} }

  files.forEach((file) => {
    // 파일명만 사용 (경로 제거)
    const fileName = file.filename || file.id
    const cleanName = fileName.split('/').pop() || fileName

    if (!root.children) root.children = {}
    root.children[cleanName] = {
      name: cleanName,
      type: "file",
      file,
    }
  })

  return root
}

const getNodeFromPath = (root: FileTreeNode, path: string[]): FileTreeNode => {
  let node: FileTreeNode = root

  for (const segment of path) {
    if (node.type !== "folder") {
      return root
    }

    const nextNode = node.children?.[segment]
    if (!nextNode) {
      return node
    }
    node = nextNode
  }

  return node
}

const listChildren = (node: FileTreeNode): FileTreeNode[] => {
  if (node.type !== "folder" || !node.children) return []
  return Object.values(node.children)
}


const formatDate = (dateString: string) => {
  if (!dateString) return "-"
  return new Date(dateString).toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  })
}

export function DataManagementDB() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [imageFiles, setImageFiles] = useState<ImageFile[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [loadingImages, setLoadingImages] = useState(false)
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null)
  const [imageDetections, setImageDetections] = useState<ImageDetection[]>([])
  const [currentPath, setCurrentPath] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [sortOption, setSortOption] = useState("recent")
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(24)

  // Use custom hook for image annotations
  const { data: annotationsData, isLoading: annotationsLoading } = useImageAnnotations(
    selectedImage?.id || '',
    { annotation_type: 'bbox', min_confidence: 0.0 },
    !!selectedImage?.id
  )

  // 동적 그리드 컬럼 계산
  const getGridCols = useMemo(() => {
    // itemsPerPage에 따라 최적의 그리드 컬럼 반환
    switch (itemsPerPage) {
      case 24:
        return "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6" // 2x3x4x6
      case 36:
        return "grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 xl:grid-cols-6" // 2x4x6x6
      default:
        return "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6"
    }
  }, [itemsPerPage])

  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [uploadForm, setUploadForm] = useState({
    name: "",
    description: "",
    imageFiles: [] as File[],
    labelFiles: [] as File[],
    classesFileObject: null as File | null,
  })
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const imagesFolderInputRef = useRef<HTMLInputElement>(null)
  const labelsFolderInputRef = useRef<HTMLInputElement>(null)
  const classesFileInputRef = useRef<HTMLInputElement>(null)

  const fetchDatasets = async () => {
    try {
      setLoadingDatasets(true)
      // Add timestamp to prevent caching
      const timestamp = new Date().getTime()
      const response = await fetch(`/api/datasets?type=2D_IMAGE&_t=${timestamp}`, {
        cache: "no-store",
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      })

      if (!response.ok) {
        throw new Error("데이터셋을 불러오지 못했습니다")
      }

      const data = await response.json()
      setDatasets(data)
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingDatasets(false)
    }
  }

  const fetchImageFiles = async (datasetId: string, page: number, pageSize: number) => {
    console.log("[DataManagementDB] Fetching images for dataset:", datasetId)
    try {
      setLoadingImages(true)
      const skip = (page - 1) * pageSize
      const response = await fetch(`/api/datasets/${datasetId}/images?skip=${skip}&limit=${pageSize}`, {
        cache: "no-store",
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      })

      if (!response.ok) {
        throw new Error("이미지 정보를 불러오지 못했습니다")
      }

      const data = await response.json()

      // API returns { total, images: [] }
      const rawImages = Array.isArray(data) ? data : (data.images || data.items || [])
      const fallbackTotal = selectedDataset?.size ?? rawImages.length
      const total = typeof data.total === "number" ? data.total : fallbackTotal

      // Convert snake_case API response to camelCase
      const images = rawImages.map((img: any) => ({
        id: img.id,
        datasetId: img.dataset_id,
        filename: img.file_name || img.filename,
        storage_key: img.storage_key,
        previewUrl: img.previewUrl || (img.storage_key ? `/api/storage/${img.storage_key}` : null),
        width: img.width,
        height: img.height,
        format: img.format || 'jpeg',
        mimeType: img.mime_type || img.mimeType || 'image/jpeg',
        metadata: img.metadata || img.metadata_,
        createdAt: img.created_at,
        detections: img.detections,
        visualization: img.visualization,
        data: img.data
      }))

      console.log("[DataManagementDB] Received image data:", {
        imagesCount: images.length,
        firstImage: images[0] ? {
          filename: images[0].filename,
          previewUrl: images[0].previewUrl,
          storage_key: images[0].storage_key,
          hasStorageKey: !!images[0].storage_key
        } : null
      })
      setImageFiles(images)
      setTotalImages(total)
    } catch (error) {
      console.error("[DataManagementDB] Error fetching images:", error)
      setImageFiles([])
      setTotalImages(0)
    } finally {
      setLoadingImages(false)
    }
  }

  useEffect(() => {
    fetchDatasets()
  }, [])

  const selectedDatasetId = selectedDataset?.id ?? ""
  const pathKey = useMemo(() => currentPath.join("/"), [currentPath])

  useEffect(() => {
    if (!selectedDataset) {
      setImageFiles([])
      setTotalImages(0)
    }
    setCurrentPath([])
    setSelectedImage(null)
    setImageDetections([])
  }, [selectedDatasetId])

  useEffect(() => {
    if (!selectedDatasetId) return
    fetchImageFiles(selectedDatasetId, currentPage, itemsPerPage)
  }, [selectedDatasetId, currentPage, itemsPerPage])

  const metadataMap = useMemo(() => {
    const map = new Map<string, ModelImageMetadata>()

    // 이미지 파일의 detections 필드에서 메타데이터 추출 (데이터베이스에서 가져온 데이터)
    imageFiles.forEach((imageFile) => {
      if (imageFile.detections && Array.isArray(imageFile.detections)) {
        const modelMetadata: ModelImageMetadata = {
          filename: imageFile.filename,
          visualization: imageFile.visualization || undefined,
          detections: imageFile.detections as ImageDetection[]
        }

        map.set(imageFile.filename, modelMetadata)

        const baseFilename = imageFile.filename.split("/").pop() || imageFile.filename
        map.set(baseFilename, modelMetadata)
      }
    })

    return map
  }, [imageFiles])

  const fileTree = useMemo(() => buildFileTree(imageFiles), [imageFiles])
  const currentNode = useMemo(
    () => getNodeFromPath(fileTree, currentPath),
    [fileTree, currentPath],
  )

  const filteredDatasets = useMemo(() => {
    if (!searchQuery) return datasets
    const lower = searchQuery.toLowerCase()
    return datasets.filter((dataset) =>
      dataset.name.toLowerCase().includes(lower) ||
      dataset.description?.toLowerCase().includes(lower),
    )
  }, [datasets, searchQuery])

  const currentItems: DisplayItem[] = useMemo(() => {
    if (!selectedDataset) {
      return filteredDatasets.map((dataset) => ({
        type: "dataset",
        dataset,
      }))
    }

    const children = listChildren(currentNode)

    const items: DisplayItem[] = children
      .filter(child => child.type === "file")
      .map((child) => ({
        type: "file",
        file: child.file!,
        metadata:
          metadataMap.get(child.file?.filename || "") ||
          metadataMap.get(child.file?.filename || "") ||
          metadataMap.get(child.name),
      }))

    const lowerQuery = searchQuery.toLowerCase()

    const filtered = searchQuery
      ? items.filter((item) => {
          if (item.type === "file") {
            const base =
              item.file.filename ||
              item.file.filename ||
              ""
            return base.toLowerCase().includes(lowerQuery)
          }
          return true
        })
      : items

    const sorted = [...filtered]

    sorted.sort((a, b) => {
      if (sortOption === "name") {
        const aName =
          a.type === "dataset"
            ? a.dataset.name
            : a.file.filename
        const bName =
          b.type === "dataset"
            ? b.dataset.name
            : b.file.filename
        return aName.localeCompare(bName, "ko")
      }

      // recent
      const aDate =
        a.type === "dataset"
          ? new Date(a.dataset.updatedAt).getTime()
          : a.type === "file"
          ? new Date(a.file.createdAt).getTime()
          : 0
      const bDate =
        b.type === "dataset"
          ? new Date(b.dataset.updatedAt).getTime()
          : b.type === "file"
          ? new Date(b.file.createdAt).getTime()
          : 0
      return bDate - aDate
    })

    return sorted
  }, [filteredDatasets, selectedDataset, currentNode, searchQuery, sortOption, metadataMap])

  useEffect(() => {
    setCurrentPage(1)
  }, [selectedDatasetId, pathKey, searchQuery, sortOption, itemsPerPage])

  const totalItems = selectedDataset
    ? Math.max(totalImages, selectedDataset.size ?? 0, currentItems.length)
    : currentItems.length
  const totalPages = Math.max(1, Math.ceil(totalItems / itemsPerPage))

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages)
    }
  }, [currentPage, totalPages])

  const paginatedItems = useMemo(() => {
    if (selectedDataset) {
      return currentItems
    }
    const start = (currentPage - 1) * itemsPerPage
    return currentItems.slice(start, start + itemsPerPage)
  }, [currentItems, currentPage, itemsPerPage, selectedDataset])

  const startIndex = totalItems === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1
  const endIndex = totalItems === 0 ? 0 : Math.min(totalItems, (currentPage - 1) * itemsPerPage + paginatedItems.length)

  const handleItemsPerPageChange = (value: string) => {
    const next = Number(value)
    if (!Number.isNaN(next) && next > 0) {
      setItemsPerPage(next)
      setCurrentPage(1)
    }
  }


  const selectedImageMetadata = useMemo(() => {
    if (!selectedImage) return null

    console.log("[DataManagementDB] Looking for metadata:", {
      selectedImage: {
        filename: selectedImage.filename,
        hasDetections: !!selectedImage.detections,
        fetchedDetectionsCount: imageDetections.length
      },
      metadataMapSize: metadataMap.size,
      metadataMapKeys: Array.from(metadataMap.keys())
    })

    // First priority: Use fetched detections from backend
    if (imageDetections && imageDetections.length > 0) {
      console.log("[DataManagementDB] Using fetched detections from backend")
      return {
        filename: selectedImage.filename,
        detections: imageDetections,
        visualization: selectedImage.visualization || undefined
      }
    }

    // Second priority: Check metadataMap
    const keyOptions = [
      selectedImage.filename,
      selectedImage.filename,
      selectedImage.filename?.split("/").pop() || "",
    ]

    for (const key of keyOptions) {
      if (!key) continue
      const meta = metadataMap.get(key)
      if (meta) {
        console.log("[DataManagementDB] Found metadata for key:", key, meta)
        return meta
      }
    }

    // Third priority: Check direct detections field
    if (selectedImage.detections) {
      console.log("[DataManagementDB] Using direct detections from selectedImage")
      return {
        filename: selectedImage.filename,
        detections: selectedImage.detections as ImageDetection[],
        visualization: selectedImage.visualization || undefined
      }
    }

    console.log("[DataManagementDB] No metadata found for selected image")
    return null
  }, [metadataMap, selectedImage, imageDetections])

  // Process annotations when data changes
  useEffect(() => {
    if (!selectedImage?.id) {
      setImageDetections([])
      return
    }

    if (annotationsLoading) return

    if (!annotationsData) {
      setImageDetections([])
      return
    }

    console.log("[DataManagementDB] Processing annotations for image:", selectedImage.id)
    console.log("[DataManagementDB] Annotations data:", annotationsData)

    const annotations = annotationsData as any[]

    if (annotations && annotations.length > 0) {
      // Get image dimensions for coordinate conversion
      const imgWidth = selectedImage.width || 1
      const imgHeight = selectedImage.height || 1

      // Convert annotations from YOLO normalized format to pixel coordinates
      // Backend stores: bbox_x (x_center), bbox_y (y_center), bbox_width, bbox_height (all normalized 0-1)
      // Convert to: x1, y1, x2, y2 (pixel coordinates)
      const detections: ImageDetection[] = annotations.map((ann: any) => {
        const xCenter = parseFloat(ann.bbox_x || 0)
        const yCenter = parseFloat(ann.bbox_y || 0)
        const width = parseFloat(ann.bbox_width || 0)
        const height = parseFloat(ann.bbox_height || 0)

        // Convert from YOLO normalized to pixel xyxy format
        const x1 = (xCenter - width / 2) * imgWidth
        const y1 = (yCenter - height / 2) * imgHeight
        const x2 = (xCenter + width / 2) * imgWidth
        const y2 = (yCenter + height / 2) * imgHeight

        // Check if this is a ground truth label (from YOLO label file)
        const isGroundTruth = ann.metadata_?.source === "YOLO label" || ann.metadata?.source === "YOLO label"

        return {
          bbox: { x1, y1, x2, y2 },
          class: ann.class_name,
          class_id: ann.class_index || 0,
          confidence: parseFloat(ann.confidence || 0),
          isGroundTruth,
        }
      })

      setImageDetections(detections)
    } else {
      setImageDetections([])
    }
  }, [selectedImage?.id, selectedImage?.width, selectedImage?.height, annotationsData, annotationsLoading])

  // const totalFiles = datasets.reduce((acc, dataset) => acc + dataset.size, 0)
  // const totalStorage = datasets.reduce(
  //   (acc, dataset) => acc + (dataset.metadata?.totalSizeBytes || 0),
  //   0,
  // )

  const resetUploadForm = () => {
    setUploadForm({
      name: "",
      description: "",
      imageFiles: [],
      labelFiles: [],
      classesFileObject: null,
    })
  }


  const handleImagesFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length > 0) {
      // Filter for image files only
      const imageFiles = files.filter(f =>
        f.type.startsWith('image/') ||
        /\.(jpg|jpeg|png|bmp|gif|webp)$/i.test(f.name)
      )

      setUploadForm((prev) => ({
        ...prev,
        imageFiles: imageFiles,
      }))
    }
  }

  const handleLabelsFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length > 0) {
      // Filter for .txt label files only
      const labelFiles = files.filter(f =>
        f.name.endsWith('.txt') && f.type === 'text/plain'
      )

      setUploadForm((prev) => ({
        ...prev,
        labelFiles: labelFiles,
      }))
    }
  }

  const handleClassesFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length > 0) {
      const file = files[0]

      setUploadForm((prev) => ({
        ...prev,
        classesFileObject: file,
      }))
    }
  }

  const handleUploadDataset = async () => {
    if (!uploadForm.name) {
      alert("데이터셋 이름을 입력해주세요.")
      return
    }

    if (!validateName(uploadForm.name)) {
      alert(getNameValidationMessage("데이터셋 이름"))
      return
    }

    // Check for duplicate dataset names
    const existingNames = datasets.map(d => d.name)
    if (isNameDuplicate(uploadForm.name, existingNames)) {
      alert(getDuplicateNameMessage("데이터셋"))
      return
    }

    if (uploadForm.imageFiles.length === 0 || uploadForm.labelFiles.length === 0) {
      alert("이미지 폴더와 라벨 폴더를 선택해주세요.")
      return
    }

    if (!uploadForm.classesFileObject) {
      alert("클래스 파일(classes.txt)을 선택해주세요.")
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const progressTimer = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 5, 90))
      }, 200)

      // Create FormData with actual files
      const formData = new FormData()
      formData.append("name", uploadForm.name)
      formData.append("description", uploadForm.description || "")

      // Add image files
      uploadForm.imageFiles.forEach((file) => {
        formData.append("image_files", file)
      })

      // Add label files
      uploadForm.labelFiles.forEach((file) => {
        formData.append("label_files", file)
      })

      // Add classes file (required)
      formData.append("classes_file", uploadForm.classesFileObject!)

      // Use Next.js API proxy instead of direct backend access
      const response = await fetch(`/api/datasets/upload-yolo`, {
        method: "POST",
        body: formData,
      })

      clearInterval(progressTimer)
      setUploadProgress(100)

      if (response.ok) {
        const result = await response.json()
        console.log("[DataManagement] Upload result:", result)
        await fetchDatasets()
        resetUploadForm()
        setShowUploadDialog(false)
        alert(`데이터셋 "${uploadForm.name}" 업로드가 완료되었습니다.\n이미지 수: ${result.image_count || result.size || uploadForm.imageFiles.length}`)
      } else {
        const error = await response.json()
        alert(`업로드 실패: ${error.detail || error.error || '알 수 없는 오류'}`)
      }
    } catch (error) {
      console.error(error)
      alert("데이터셋 업로드 중 오류가 발생했습니다.")
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  const handleDatasetClick = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    setTotalImages(dataset.size ?? 0)
    setCurrentPage(1)
  }


  const handleDeleteDataset = async (datasetId: string, datasetName: string) => {
    if (!confirm(`"${datasetName}" 데이터셋을 삭제하시겠습니까?\n모든 이미지 파일도 함께 삭제됩니다.`)) {
      return
    }

    try {
      const response = await fetch(`/api/datasets/${datasetId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        await fetchDatasets()
        if (selectedDataset?.id === datasetId) {
          setSelectedDataset(null)
          setImageFiles([])
          setSelectedImage(null)
        }
        alert('데이터셋이 삭제되었습니다.')
      } else {
        const error = await response.json()
        alert(`삭제 실패: ${error.error}`)
      }
    } catch (error) {
      console.error('Delete error:', error)
      alert('삭제 중 오류가 발생했습니다.')
    }
  }

  const handleDeleteImage = async (imageId: string, imageName: string) => {
    if (!confirm(`"${imageName}" 이미지를 삭제하시겠습니까?`)) {
      return
    }

    try {
      const response = await fetch(`/api/datasets/images/${imageId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        // Clear selected image immediately
        if (selectedImage?.id === imageId) {
          setSelectedImage(null)
          setImageDetections([])
        }

        // Refresh data
        if (selectedDataset) {
          await fetchImageFiles(selectedDataset.id, currentPage, itemsPerPage)
          await fetchDatasets() // 데이터셋의 파일 개수 업데이트
        }

        alert('이미지가 삭제되었습니다.')
      } else {
        const error = await response.json()
        alert(`삭제 실패: ${error.error}`)
      }
    } catch (error) {
      console.error('Delete error:', error)
      alert('삭제 중 오류가 발생했습니다.')
    }
  }

  const handleRefresh = () => {
    if (selectedDataset) {
      fetchImageFiles(selectedDataset.id, currentPage, itemsPerPage)
    } else {
      fetchDatasets()
    }
  }

  return (
    <>
      {/* Hidden file inputs for folder selection */}
      <input
        ref={imagesFolderInputRef}
        type="file"
        // @ts-expect-error - webkitdirectory is not in TypeScript types
        webkitdirectory="true"
        multiple
        onChange={handleImagesFolderSelect}
        className="hidden"
      />
      <input
        ref={labelsFolderInputRef}
        type="file"
        // @ts-expect-error - webkitdirectory is not in TypeScript types
        webkitdirectory="true"
        multiple
        onChange={handleLabelsFolderSelect}
        className="hidden"
      />
      <input
        ref={classesFileInputRef}
        type="file"
        accept=".txt"
        onChange={handleClassesFileSelect}
        className="hidden"
      />

      <AdversarialToolLayout
        title="2D 이미지 데이터 관리"
        description="데이터셋과 이미지 파일을 탐색하고 미리보기와 메타데이터를 확인하세요"
        icon={Database}
        headerStats={
          <div className="flex flex-wrap items-center gap-2">
            <Button
              onClick={() => setShowUploadDialog(true)}
              className="ds-btn-primary"
            >
              <Upload className="mr-2 h-4 w-4" /> 폴더 업로드
            </Button>
            <Button
              onClick={handleRefresh}
              className="ds-btn-outline"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> 새로고침
            </Button>
          </div>
        }
        reversePanels={true}
        leftPanel={{
          title: "데이터셋 & 이미지",
          icon: LayoutGrid,
          description: "데이터셋을 선택하고 이미지를 탐색하세요",
          children: (
            <div className="flex flex-col h-full">
              <div className="flex-shrink-0 flex flex-col gap-2 sm:gap-3 sm:flex-row sm:items-center sm:justify-between pb-3 sm:pb-4">
                <div className="flex items-center gap-2 flex-1">
                  {selectedDataset && (
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => {
                        setSelectedDataset(null)
                        setImageFiles([])
                        setSelectedImage(null)
                      }}
                      className="border-outline bg-surface-container-high text-muted hover:bg-surface-container flex-shrink-0 h-9 w-9 sm:h-10 sm:w-10"
                      title="데이터셋 목록으로 돌아가기"
                    >
                      <ChevronLeft className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                    </Button>
                  )}
                  <div className="relative flex-1 max-w-xs">
                    <Search className="absolute left-2.5 sm:left-3 top-2.5 sm:top-3 h-3.5 w-3.5 sm:h-4 sm:w-4 text-muted" />
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder="이름으로 검색"
                      className="w-full h-9 sm:h-10 border-outline-variant bg-surface-container pl-9 sm:pl-10 text-sm text-foreground placeholder:text-muted"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap sm:flex-nowrap">
                  <Select value={itemsPerPage.toString()} onValueChange={handleItemsPerPageChange}>
                    <SelectTrigger className="w-[80px] sm:min-w-[100px] h-9 sm:h-10 border-outline-variant bg-surface-container text-xs sm:text-sm text-foreground">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="24">24개</SelectItem>
                      <SelectItem value="36">36개</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={sortOption} onValueChange={setSortOption}>
                    <SelectTrigger className="w-[100px] sm:min-w-[140px] h-9 sm:h-10 border-outline-variant bg-surface-container text-xs sm:text-sm text-foreground">
                      <SelectValue placeholder="정렬" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="recent">최신순</SelectItem>
                      <SelectItem value="name">이름순</SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="flex overflow-hidden rounded-md border border-outline-variant bg-surface-container-low">
                    <Button
                      size="icon"
                      variant={viewMode === "grid" ? "secondary" : "ghost"}
                      onClick={() => setViewMode("grid")}
                      className="h-9 w-9 sm:h-10 sm:w-10"
                    >
                      <LayoutGrid className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant={viewMode === "list" ? "secondary" : "ghost"}
                      onClick={() => setViewMode("list")}
                      className="h-9 w-9 sm:h-10 sm:w-10"
                    >
                      <List className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex-1 min-h-0 overflow-y-auto pb-2">
                {loadingDatasets || loadingImages ? (
                  <div className="flex h-64 items-center justify-center text-muted">
                    <RefreshCw className="h-6 w-6 animate-spin" />
                  </div>
                ) : currentItems.length === 0 ? (
                  <div className="flex h-64 flex-col items-center justify-center gap-3 text-center text-muted">
                    <FolderOpen className="h-12 w-12 text-muted" />
                    <p>표시할 항목이 없습니다.</p>
                  </div>
                ) : viewMode === "grid" ? (
                  <div className={`grid gap-2 sm:gap-3 lg:gap-4 ${getGridCols} auto-rows-max`}>
                    {paginatedItems.map((item) => {
                      if (item.type === "dataset") {
                        const { dataset } = item
                        return (
                          <div
                            key={dataset.id}
                            className="group relative rounded-xl border border-outline-variant bg-surface-container p-3 sm:p-4 lg:p-5 transition hover:-translate-y-1 hover:border-primary hover:bg-surface-container-high"
                          >
                            <button
                              onClick={() => handleDatasetClick(dataset)}
                              className="w-full text-left"
                            >
                              <div className="flex items-start gap-2">
                                <Folder className="h-6 w-6 sm:h-7 sm:w-7 lg:h-8 lg:w-8 text-primary flex-shrink-0" />
                              </div>
                              <h3 className="mt-3 sm:mt-4 text-sm sm:text-base lg:text-lg font-semibold text-foreground line-clamp-2 break-words">
                                {dataset.name}
                              </h3>
                              {dataset.description && (
                                <p className="mt-1 sm:mt-2 line-clamp-2 text-xs sm:text-sm text-muted break-words">
                                  {dataset.description}
                                </p>
                              )}
                              <Separator className="my-2 sm:my-3 lg:my-4 bg-surface-container/80" />
                              <div className="flex flex-wrap gap-1 sm:gap-2 text-[10px] sm:text-xs text-muted">
                                <span className="truncate">{dataset.size.toLocaleString()} 파일</span>
                                <span>•</span>
                                <span className="truncate">{formatDate(dataset.updatedAt)}</span>
                              </div>
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDeleteDataset(dataset.id, dataset.name)
                              }}
                              className="absolute top-2 right-2 sm:top-3 sm:right-3 p-1.5 sm:p-2 rounded-lg bg-error-container hover:bg-error-container transition-colors"
                            >
                              <Trash2 className="h-3 w-3 sm:h-4 sm:w-4 text-error" />
                            </button>
                          </div>
                        )
                      }


                      const { file, metadata } = item
                      const displayName = file.filename
                      const detections = metadata?.detections.length ?? 0
                      const isSelected = selectedImage?.id === file.id

                      return (
                        <div
                          key={file.id}
                          className={`relative rounded-xl border p-3 sm:p-4 lg:p-5 transition ${
                            isSelected
                              ? "border-tertiary bg-tertiary-container text-on-tertiary-container shadow-lg"
                              : "border-outline-variant bg-surface-container hover:-translate-y-1 hover:border-primary hover:bg-surface-container-high"
                          }`}>
                          {isSelected && (
                            <div className="absolute top-0 left-0 w-full h-1 bg-tertiary rounded-t-xl" />
                          )}
                          <button
                            onClick={() => {
                              console.log("[DataManagementDB] Selected image:", {
                                filename: file.filename,
                                hasData: !!file.data,
                                dataLength: file.data?.length,
                                storage_key: file.storage_key,
                                mimeType: file.mimeType
                              })
                              setSelectedImage(file)
                            }}
                            className="w-full text-left"
                          >
                            <div className="flex items-start justify-between">
                              <div className={`flex size-10 sm:size-11 lg:size-12 items-center justify-center rounded-lg ${
                                isSelected ? "bg-tertiary/30" : "bg-primary/10"
                              }`}>
                                <FileImage className={`h-5 w-5 sm:h-6 sm:w-6 ${isSelected ? "text-tertiary" : "text-primary-foreground"}`} />
                              </div>
                              {detections > 0 && (
                                <Badge variant="outline" className={`text-[10px] sm:text-xs ${
                                  isSelected
                                    ? "border-primary/60 bg-primary/20 text-primary-foreground"
                                    : "border-primary bg-primary/10 text-primary-foreground"
                                }`}>
                                  탐지 {detections}
                                </Badge>
                              )}
                            </div>
                            <h3 className={`mt-3 sm:mt-4 line-clamp-2 text-xs sm:text-sm lg:text-base font-semibold break-words ${
                              isSelected ? "text-on-tertiary-container" : "text-foreground"
                            }`}>
                              {displayName}
                            </h3>
                            <Separator className={`my-2 sm:my-3 ${isSelected ? "bg-outline" : "bg-outline-variant"}`} />
                            <div className="flex flex-wrap gap-1 sm:gap-2 text-[10px] sm:text-xs text-muted">
                              <span className="truncate">{file.format}</span>
                              {file.width && file.height && (
                                <>
                                  <span>•</span>
                                  <span className="truncate">{file.width}×{file.height}</span>
                                </>
                              )}
                            </div>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteImage(file.id, file.filename)
                            }}
                            className="absolute top-2 right-2 sm:top-3 sm:right-3 p-1.5 sm:p-2 rounded-lg bg-error-container hover:bg-error-container transition-colors"
                          >
                            <Trash2 className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-error" />
                          </button>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {paginatedItems.map((item) => {
                      if (item.type === "dataset") {
                        const { dataset } = item
                        return (
                          <button
                            key={dataset.id}
                            onClick={() => handleDatasetClick(dataset)}
                            className="flex items-center justify-between gap-4 rounded-lg border border-outline-variant bg-surface-container px-4 py-3 text-left transition hover:border-primary hover:bg-surface-container-high"
                          >
                            <div className="flex items-center gap-3 min-w-0 flex-1">
                              <Folder className="h-6 w-6 text-primary flex-shrink-0" />
                              <div className="min-w-0 flex-1">
                                <p className="font-semibold text-foreground truncate">{dataset.name}</p>
                                <p className="text-xs text-muted">{dataset.size.toLocaleString()} 파일</p>
                              </div>
                            </div>
                            <span className="text-xs text-muted whitespace-nowrap flex-shrink-0">{formatDate(dataset.updatedAt)}</span>
                          </button>
                        )
                      }


                      const { file, metadata } = item
                      const isSelected = selectedImage?.id === file.id
                      return (
                        <div key={file.id} className="relative">
                          {isSelected && (
                            <div className="absolute top-0 left-0 right-0 h-1 bg-tertiary rounded-t-lg" />
                          )}
                          <button
                            onClick={() => {
                              console.log("[DataManagementDB] Selected image (list view):", {
                                filename: file.filename,
                                hasData: !!file.data,
                                dataLength: file.data?.length,
                                mimeType: file.mimeType
                              })
                              setSelectedImage(file)
                            }}
                            className={`relative flex items-center justify-between gap-4 rounded-lg border px-4 py-3 text-left transition w-full ${
                              isSelected
                                ? "border-tertiary bg-tertiary-container shadow-lg"
                                : "border-outline-variant bg-surface-container hover:border-primary hover:bg-surface-container-high"
                            }`}>
                                                  {isSelected && (
                                                    <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-tertiary rounded-l-lg" />
                                                  )}                            <div className="flex items-center gap-3 min-w-0 flex-1">
                              <div className={`flex h-10 w-10 items-center justify-center rounded-lg flex-shrink-0 ${
                                isSelected ? "bg-tertiary" : "bg-primary-container"
                              }`}>
                                <FileImage className={`h-5 w-5 ${isSelected ? "text-on-tertiary-container" : "text-on-primary-container"}`} />
                              </div>
                              <div className="min-w-0 flex-1">
                                <p className={`text-sm font-semibold truncate ${isSelected ? "text-on-tertiary-container" : "text-foreground"}`}>{file.filename}</p>
                                <div className="flex flex-wrap gap-2 text-xs text-muted">
                                  <span>{file.format}</span>
                                  {(metadata?.detections.length ?? 0) > 0 && (
                                    <>
                                      <span>•</span>
                                      <span>탐지 {metadata?.detections.length ?? 0}개</span>
                                    </>
                                  )}
                                  {file.width && file.height && (
                                    <>
                                      <span>•</span>
                                      <span>{file.width}×{file.height}</span>
                                    </>
                                  )}
                                </div>
                              </div>
                            </div>
                            <span className="text-xs text-muted whitespace-nowrap flex-shrink-0">{formatDate(file.createdAt)}</span>
                          </button>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-outline-variant bg-surface-container-high px-3 sm:px-4 py-1.5 sm:py-2 flex-shrink-0 mt-2 sm:mt-3">
            <div className="flex flex-col gap-2 sm:gap-3 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:text-sm">
              <span className="text-center sm:text-left">
                총 {totalItems.toLocaleString()}개 중 {startIndex}-{endIndex} 표시
              </span>
              <div className="flex items-center gap-2 justify-center sm:justify-end">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                  className="border-outline bg-surface-container-high text-muted hover:bg-background h-8 w-8 sm:h-9 sm:w-9"
                  aria-label="이전 페이지"
                >
                  <ChevronLeft className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                </Button>
                <span className="min-w-[60px] sm:min-w-[72px] text-center text-muted text-xs sm:text-sm">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages || currentItems.length === 0}
                  className="border-outline bg-surface-container-high text-muted hover:bg-background h-8 w-8 sm:h-9 sm:w-9"
                  aria-label="다음 페이지"
                >
                  <ChevronRight className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                </Button>
              </div>
              </div>
            </div>
          </div>
          )
        }}
        rightPanel={{
          title: "이미지 미리보기",
          icon: Eye,
          description: "이미지를 선택하면 메타데이터와 함께 미리보기가 표시됩니다",
          children: (
            <div className="flex flex-col space-y-4 h-full">
              {selectedImage ? (
                <div className="flex flex-col space-y-4 overflow-y-auto">
                  <div className="flex-shrink-0">
                    <div className="mb-3 pb-3 border-b border-outline-variant">
                      <div className="flex items-center gap-2">
                        <FileImage className="h-5 w-5 text-primary flex-shrink-0" />
                        <h3 className="text-base font-semibold text-foreground break-words">{selectedImage.filename}</h3>
                      </div>
                      {selectedImage.width && selectedImage.height && (
                        <p className="text-xs text-muted mt-1 ml-7">
                          {selectedImage.width} × {selectedImage.height} px · {selectedImage.format.toUpperCase()}
                        </p>
                      )}
                    </div>
                  {selectedImage.previewUrl || selectedImage.storage_key || selectedImage.data ? (
                    <AnnotatedImageViewer
                      imageId={selectedImage.id}
                      imageUrl={
                        selectedImage.previewUrl
                          ? selectedImage.previewUrl
                          : selectedImage.storage_key
                          ? `/api/storage/${selectedImage.storage_key}`
                          : `data:${selectedImage.mimeType || 'image/jpeg'};base64,${selectedImage.data}`
                      }
                        minConfidence={0.0}
                        showLabels={true}
                        className="w-full rounded-none border-0"
                      />
                    ) : (
                      <div className="flex h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-outline bg-surface-container-low text-muted">
                        <ImageIcon className="h-12 w-12" />
                        <span className="text-sm">미리보기를 불러올 수 없습니다.</span>
                      </div>
                    )}
                  </div>

                  <Separator className="bg-surface-container/80" />

                  {annotationsLoading ? (
                    <Alert className="border-outline/80 bg-surface-container text-muted">
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      <AlertDescription>라벨링 정보를 불러오는 중입니다...</AlertDescription>
                    </Alert>
                  ) : selectedImageMetadata ? (
                    <div className="space-y-4">
                      <div>
                        <h4 className="flex items-center gap-2 text-sm font-semibold text-foreground"><Info className="h-4 w-4 text-primary" /> 라벨링 정보</h4>
                        {/* 탐지된 모든 클래스 표시 */}
                        {selectedImageMetadata.detections.length > 0 && (
                          <div className="mt-2">
                            <p className="text-xs text-muted mb-1">라벨링된 클래스:</p>
                            <div className="flex flex-wrap gap-1">
                              {Array.from(new Set(selectedImageMetadata.detections.map(d => d.class))).map(className => (
                                <Badge key={className} variant="outline" className="border-tertiary bg-tertiary-container text-on-tertiary-container-foreground text-xs">{className}</Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted">탐지 개수</span>
                        <span className="font-semibold text-foreground">{selectedImageMetadata.detections.length}</span>
                      </div>
                      <div className="max-h-64 overflow-y-auto pr-2">
                        <div className="grid grid-cols-2 gap-2">
                          {selectedImageMetadata.detections.map((detection, index) => (
                            <div key={`${selectedImage.id}-det-${index}`} className="rounded-lg border border-outline-variant bg-surface-container-high p-3">
                              <div className="flex items-center gap-2 mb-2">
                                <Badge variant="primary" className="border-primary text-primary-foreground text-xs">{detection.class}</Badge>
                                {!detection.isGroundTruth && (
                                  <span className="text-xs text-muted">{(detection.confidence * 100).toFixed(1)}%</span>
                                )}
                              </div>
                              <div className="text-xs text-muted font-mono">
                                <div>X: {detection.bbox.x1.toFixed(0)} - {detection.bbox.x2.toFixed(0)}</div>
                                <div>Y: {detection.bbox.y1.toFixed(0)} - {detection.bbox.y2.toFixed(0)}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Alert className="border-outline/80 bg-surface-container text-muted">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>선택한 이미지에 대한 detection 정보를 찾을 수 없습니다.</AlertDescription>
                    </Alert>
                  )}
                </div>
              ) : (
                <div className="flex h-[520px] flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-outline-variant bg-surface-container-low text-center text-muted">
                  <ImageIcon className="h-12 w-12" />
                  <div>
                    <p className="text-sm font-medium text-muted">이미지를 선택해주세요</p>
                    <p className="mt-1 text-xs text-muted">이미지 카드 중 하나를 클릭하면 이곳에 미리보기와 메타데이터가 표시됩니다.</p>
                  </div>
                </div>
              )}
            </div>
          )
        }}
      />

      <Dialog open={showUploadDialog} onOpenChange={(open) => {
        setShowUploadDialog(open)
        if (!open) {
          resetUploadForm()
        }
      }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>데이터셋 등록</DialogTitle>
            <DialogDescription className="text-muted">
              YOLO 형식 데이터셋의 경로를 입력하여 데이터베이스에 등록합니다.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <Alert className="bg-primary-container border-primary">
              <Info className="h-4 w-4 text-primary" />
              <AlertDescription className="text-muted text-xs">
                로컬 컴퓨터의 YOLO 데이터셋 폴더를 선택하여 서버로 업로드합니다.<br/>
                폴더 버튼을 클릭하여 이미지 폴더, 라벨 폴더, 클래스 파일을 선택하세요.
              </AlertDescription>
            </Alert>

            <div>
              <Label className="text-foreground">데이터셋 이름 *</Label>
              <Input
                value={uploadForm.name}
                onChange={(event) => {
                  const value = event.target.value
                  if (value === '' || validateName(value)) {
                    setUploadForm((prev) => ({ ...prev, name: value }))
                  }
                }}
                placeholder="예: COCO_Person_100"
                className="mt-1 border-outline-variant bg-surface-container text-foreground placeholder:text-muted"
                disabled={isUploading}
              />
              <p className="text-xs text-muted mt-1">영문자, 숫자, - (대시), _ (언더스코어)만 사용 가능</p>
            </div>

            <div>
              <Label className="text-foreground">이미지 폴더 *</Label>
              <div className="flex gap-2 mt-1">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => imagesFolderInputRef.current?.click()}
                  className="flex-shrink-0 border-outline-variant bg-surface-container text-foreground hover:bg-surface-container"
                  disabled={isUploading}
                >
                  <FolderOpen className="w-4 h-4 mr-2" />
                  폴더 선택
                </Button>
                <Input
                  value={uploadForm.imageFiles.length > 0 ? `${uploadForm.imageFiles.length}개 파일 선택됨` : ""}
                  readOnly
                  placeholder="이미지 폴더를 선택하세요"
                  className={`flex-1 bg-surface-container font-mono text-sm ${
                    uploadForm.imageFiles.length > 0
                      ? "border-tertiary text-tertiary placeholder:text-tertiary/50"
                      : "border-outline text-foreground placeholder:text-muted"
                  }`}
                  disabled={isUploading}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center gap-2 mb-1">
                <Label className="text-foreground">라벨 폴더 *</Label>
                <span className="text-xs text-muted">(객체의 위치 정보)</span>
              </div>
              <p className="text-xs text-muted mb-2">
                각 이미지에서 탐지된 객체의 <strong className="text-muted">위치(바운딩 박스)</strong>와 <strong className="text-muted">클래스 ID</strong>가 포함된 .txt 파일들
              </p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => labelsFolderInputRef.current?.click()}
                  className="flex-shrink-0 border-outline-variant bg-surface-container text-foreground hover:bg-surface-container"
                  disabled={isUploading}
                >
                  <FolderOpen className="w-4 h-4 mr-2" />
                  폴더 선택
                </Button>
                <Input
                  value={uploadForm.labelFiles.length > 0 ? `${uploadForm.labelFiles.length}개 파일 선택됨` : ""}
                  readOnly
                  placeholder="라벨 폴더를 선택하세요"
                  className={`flex-1 bg-surface-container font-mono text-sm ${
                    uploadForm.labelFiles.length > 0
                      ? "border-tertiary text-tertiary placeholder:text-tertiary/50"
                      : "border-outline text-foreground placeholder:text-muted"
                  }`}
                  disabled={isUploading}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center gap-2 mb-1">
                <Label className="text-foreground">클래스 파일 *</Label>
                <span className="text-xs text-muted">(객체의 종류 목록)</span>
              </div>
              <p className="text-xs text-muted mb-2">
                탐지 가능한 <strong className="text-muted">모든 객체 종류의 이름</strong>이 나열된 파일 (예: person, car, dog 등)
              </p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => classesFileInputRef.current?.click()}
                  className="flex-shrink-0 border-outline-variant bg-surface-container text-foreground hover:bg-surface-container"
                  disabled={isUploading}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  파일 선택
                </Button>
                <Input
                  value={uploadForm.classesFileObject ? uploadForm.classesFileObject.name : ""}
                  readOnly
                  placeholder="classes.txt 파일을 선택하세요"
                  className={`flex-1 bg-surface-container font-mono text-sm ${
                    uploadForm.classesFileObject
                      ? "border-tertiary text-tertiary placeholder:text-tertiary/50"
                      : "border-outline text-foreground placeholder:text-muted"
                  }`}
                  disabled={isUploading}
                />
              </div>
            </div>

            <div>
              <Label className="text-foreground">설명 (선택)</Label>
              <Textarea
                value={uploadForm.description}
                onChange={(event) =>
                  setUploadForm((prev) => ({ ...prev, description: event.target.value }))
                }
                rows={2}
                placeholder="데이터셋에 대한 설명을 입력하세요"
                className="mt-1 border-outline-variant bg-surface-container text-foreground placeholder:text-muted"
                disabled={isUploading}
              />
            </div>

            {isUploading && (
              <div className="space-y-2">
                <Progress value={uploadProgress} className="h-2" />
                <p className="text-xs text-muted">업로드 진행 중... {uploadProgress}%</p>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button onClick={() => setShowUploadDialog(false)} disabled={isUploading} className="ds-btn-outline">
              취소
            </Button>
            <Button onClick={handleUploadDataset} disabled={isUploading} className="ds-btn-primary">
              업로드 시작
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
