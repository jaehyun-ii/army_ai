"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { apiClient } from "@/lib/api-client"
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
  const [loadingImages, setLoadingImages] = useState(false)
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null)
  const [imageDetections, setImageDetections] = useState<ImageDetection[]>([])
  const [currentPath, setCurrentPath] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [sortOption, setSortOption] = useState("recent")
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(24)

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

  const fetchImageFiles = async (datasetId: string) => {
    console.log("[DataManagementDB] Fetching images for dataset:", datasetId)
    try {
      setLoadingImages(true)
      // Add timestamp to prevent caching
      const timestamp = new Date().getTime()
      const response = await fetch(`/api/datasets/${datasetId}/images?_t=${timestamp}`, {
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
      console.log("[DataManagementDB] Received image data:", {
        total: data.total,
        imagesCount: data.images?.length,
        firstImage: data.images?.[0] ? {
          filename: data.images[0].filename,
          hasData: !!data.images[0].data,
          dataLength: data.images[0].data?.length
        } : null
      })
      setImageFiles(data.images || [])
    } catch (error) {
      console.error("[DataManagementDB] Error fetching images:", error)
      setImageFiles([])
    } finally {
      setLoadingImages(false)
    }
  }

  useEffect(() => {
    fetchDatasets()
  }, [])

  useEffect(() => {
    if (selectedDataset) {
      fetchImageFiles(selectedDataset.id)
    } else {
      setImageFiles([])
    }
    setCurrentPath([])
    setSelectedImage(null)
  }, [selectedDataset])

  const selectedDatasetId = selectedDataset?.id ?? ""
  const pathKey = useMemo(() => currentPath.join("/"), [currentPath])

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
  }, [selectedDatasetId, pathKey, searchQuery, sortOption])

  const totalPages = Math.max(1, Math.ceil(currentItems.length / itemsPerPage))

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages)
    }
  }, [currentPage, totalPages])

  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage
    return currentItems.slice(start, start + itemsPerPage)
  }, [currentItems, currentPage, itemsPerPage])

  const startIndex = currentItems.length === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1
  const endIndex = currentItems.length === 0 ? 0 : Math.min(currentItems.length, currentPage * itemsPerPage)

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

  // Fetch annotations from backend when image is selected
  useEffect(() => {
    if (!selectedImage?.id) {
      setImageDetections([])
      return
    }

    const fetchAnnotations = async () => {
      try {
        console.log("[DataManagementDB] Fetching annotations for image:", selectedImage.id)

        const annotations = await apiClient.getImageAnnotations(selectedImage.id, {
          annotation_type: 'bbox',
          min_confidence: 0.0
        }) as any[]

        console.log("[DataManagementDB] Fetched annotations:", annotations)

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
      } catch (error) {
        console.error("[DataManagementDB] Error fetching annotations:", error)
        setImageDetections([])
      }
    }

    fetchAnnotations()
  }, [selectedImage?.id, selectedImage?.width, selectedImage?.height])

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

      const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

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

      const response = await fetch(`${BACKEND_API_URL}/api/v1/dataset-service/upload-yolo-files`, {
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
    fetchImageFiles(dataset.id)
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
          await fetchImageFiles(selectedDataset.id)
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
      fetchImageFiles(selectedDataset.id)
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
              variant="secondary"
              size="sm"
              onClick={() => setShowUploadDialog(true)}
              className="bg-blue-500/20 text-blue-200 hover:bg-blue-500/30"
            >
              <Upload className="mr-2 h-4 w-4" /> 폴더 업로드
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="border-slate-700/80 bg-slate-900/60 text-slate-200 hover:bg-slate-800"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> 새로고침
            </Button>
          </div>
        }
        leftPanelWidth="4xl"
        leftPanel={{
          title: "데이터셋 & 이미지",
          icon: LayoutGrid,
          description: "데이터셋을 선택하고 이미지를 탐색하세요",
          children: (
            <div className="flex flex-col h-full">
              <div className="flex-shrink-0 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between pb-4">
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
                      className="border-slate-800 bg-slate-900/70 text-slate-300 hover:bg-slate-800 flex-shrink-0"
                      title="데이터셋 목록으로 돌아가기"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                  )}
                  <div className="relative flex-1 max-w-xs">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder="이름으로 검색"
                      className="w-full border-slate-800 bg-slate-900/70 pl-10 text-slate-200 placeholder:text-slate-500"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Select value={itemsPerPage.toString()} onValueChange={handleItemsPerPageChange}>
                    <SelectTrigger className="min-w-[100px] border-slate-800 bg-slate-900/70 text-slate-200">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="24">24개</SelectItem>
                      <SelectItem value="36">36개</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={sortOption} onValueChange={setSortOption}>
                    <SelectTrigger className="min-w-[140px] border-slate-800 bg-slate-900/70 text-slate-200">
                      <SelectValue placeholder="정렬" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="recent">최신순</SelectItem>
                      <SelectItem value="name">이름순</SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="flex overflow-hidden rounded-md border border-slate-800">
                    <Button
                      size="icon"
                      variant={viewMode === "grid" ? "default" : "ghost"}
                      className={`${viewMode === "grid" ? "bg-blue-500/20 text-blue-200" : "text-slate-300"}`}
                      onClick={() => setViewMode("grid")}
                    >
                      <LayoutGrid className="h-4 w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant={viewMode === "list" ? "default" : "ghost"}
                      className={`${viewMode === "list" ? "bg-blue-500/20 text-blue-200" : "text-slate-300"}`}
                      onClick={() => setViewMode("list")}
                    >
                      <List className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex-1 min-h-0 overflow-y-auto pb-2">
                {loadingDatasets || loadingImages ? (
                  <div className="flex h-64 items-center justify-center text-slate-400">
                    <RefreshCw className="h-6 w-6 animate-spin" />
                  </div>
                ) : currentItems.length === 0 ? (
                  <div className="flex h-64 flex-col items-center justify-center gap-3 text-center text-slate-400">
                    <FolderOpen className="h-12 w-12 text-slate-600" />
                    <p>표시할 항목이 없습니다.</p>
                  </div>
                ) : viewMode === "grid" ? (
                  <div className={`grid gap-4 ${getGridCols} auto-rows-max`}>
                    {paginatedItems.map((item) => {
                      if (item.type === "dataset") {
                        const { dataset } = item
                        return (
                          <div
                            key={dataset.id}
                            className="group relative rounded-xl border border-slate-800/60 bg-slate-900/60 p-5 transition hover:-translate-y-1 hover:border-blue-500/50 hover:bg-slate-900/80"
                          >
                            <button
                              onClick={() => handleDatasetClick(dataset)}
                              className="w-full text-left"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <Folder className="h-8 w-8 text-blue-300 flex-shrink-0" />
                                <Badge variant="outline" className="border-slate-700/70 text-xs text-slate-300 whitespace-nowrap">
                                  {dataset.type}
                                </Badge>
                              </div>
                              <h3 className="mt-4 text-lg font-semibold text-slate-100 line-clamp-2 break-words">
                                {dataset.name}
                              </h3>
                              {dataset.description && (
                                <p className="mt-2 line-clamp-2 text-sm text-slate-400">
                                  {dataset.description}
                                </p>
                              )}
                              <Separator className="my-4 bg-slate-800/80" />
                              <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                                <span>{dataset.size.toLocaleString()} 파일</span>
                                <span>•</span>
                                <span>{formatDate(dataset.updatedAt)}</span>
                              </div>
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDeleteDataset(dataset.id, dataset.name)
                              }}
                              className="absolute bottom-3 right-3 p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 transition-colors"
                            >
                              <Trash2 className="h-4 w-4 text-red-400" />
                            </button>
                          </div>
                        )
                      }


                      const { file, metadata } = item
                      const displayName = file.filename
                      const detections = metadata?.detections.length ?? 0

                      return (
                        <div
                          key={file.id}
                          className={`relative rounded-xl border border-slate-800/60 bg-slate-900/60 p-5 transition hover:-translate-y-1 hover:border-blue-500/50 hover:bg-slate-900/80 ${selectedImage?.id === file.id ? "border-blue-500/60 bg-blue-950/40" : ""}`}>
                          <button
                            onClick={() => {
                              console.log("[DataManagementDB] Selected image:", {
                                filename: file.filename,
                                hasData: !!file.data,
                                dataLength: file.data?.length,
                                mimeType: file.mimeType
                              })
                              setSelectedImage(file)
                            }}
                            className="w-full text-left"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex size-12 items-center justify-center rounded-lg bg-blue-500/10">
                                <FileImage className="h-6 w-6 text-blue-200" />
                              </div>
                              {detections > 0 && (
                                <Badge variant="outline" className="border-blue-500/40 bg-blue-500/10 text-xs text-blue-100">
                                  탐지 {detections}
                                </Badge>
                              )}
                            </div>
                            <h3 className="mt-4 line-clamp-2 text-base font-semibold text-slate-100 break-words">
                              {displayName}
                            </h3>
                            <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
                              <span>{file.format}</span>
                              <span>•</span>
                              <span>{formatDate(file.createdAt)}</span>
                            </div>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteImage(file.id, file.filename)
                            }}
                            className="absolute bottom-3 right-3 p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 transition-colors"
                          >
                            <Trash2 className="h-3 w-3 text-red-400" />
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
                            className="flex items-center justify-between gap-4 rounded-lg border border-slate-800/60 bg-slate-900/60 px-4 py-3 text-left transition hover:border-blue-500/50 hover:bg-slate-900/80"
                          >
                            <div className="flex items-center gap-3 min-w-0 flex-1">
                              <Folder className="h-6 w-6 text-blue-300 flex-shrink-0" />
                              <div className="min-w-0 flex-1">
                                <p className="font-semibold text-slate-100 truncate">{dataset.name}</p>
                                <p className="text-xs text-slate-400">{dataset.size.toLocaleString()} 파일</p>
                              </div>
                            </div>
                            <span className="text-xs text-slate-500 whitespace-nowrap flex-shrink-0">{formatDate(dataset.updatedAt)}</span>
                          </button>
                        )
                      }


                      const { file, metadata } = item
                      return (
                        <button
                          key={file.id}
                          onClick={() => {
                            console.log("[DataManagementDB] Selected image (list view):", {
                              filename: file.filename,
                              hasData: !!file.data,
                              dataLength: file.data?.length,
                              mimeType: file.mimeType
                            })
                            setSelectedImage(file)
                          }}
                          className={`flex items-center justify-between gap-4 rounded-lg border border-slate-800/60 bg-slate-900/60 px-4 py-3 text-left transition hover:border-blue-500/50 hover:bg-slate-900/80 ${selectedImage?.id === file.id ? "border-blue-500/60 bg-blue-950/40" : ""}`}>
                          <div className="flex items-center gap-3 min-w-0 flex-1">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500/10 flex-shrink-0">
                              <FileImage className="h-5 w-5 text-blue-200" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <p className="text-sm font-semibold text-slate-100 truncate">{file.filename}</p>
                              <p className="text-xs text-slate-400">탐지 {metadata?.detections.length ?? 0}개</p>
                            </div>
                          </div>
                          <span className="text-xs text-slate-500 whitespace-nowrap flex-shrink-0">{formatDate(file.createdAt)}</span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-slate-800/60 bg-slate-900/70 px-4 py-1 flex-shrink-0 mt-3">
            <div className="flex flex-col gap-3 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between sm:text-sm">
              <span>
                총 {currentItems.length.toLocaleString()}개 중 {startIndex}-{endIndex} 표시
              </span>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                  className="border-slate-800 bg-slate-900/70 text-slate-300 hover:bg-slate-900"
                  aria-label="이전 페이지"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="min-w-[72px] text-center text-slate-300">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages || currentItems.length === 0}
                  className="border-slate-800 bg-slate-900/70 text-slate-300 hover:bg-slate-900"
                  aria-label="다음 페이지"
                >
                  <ChevronRight className="h-4 w-4" />
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
                    {selectedImage.data ? (
                      <AnnotatedImageViewer
                        imageId={selectedImage.id}
                        imageUrl={`data:${selectedImage.mimeType || 'image/jpeg'};base64,${selectedImage.data}`}
                        minConfidence={0.0}
                        showLabels={true}
                        className="w-full rounded-none border-0"
                      />
                    ) : (
                      <div className="flex h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-slate-800 bg-slate-950 text-slate-500">
                        <ImageIcon className="h-12 w-12" />
                        <span className="text-sm">미리보기를 불러올 수 없습니다.</span>
                      </div>
                    )}
                  </div>

                  <Separator className="bg-slate-800/80" />

                  {selectedImageMetadata ? (
                    <div className="space-y-4">
                      <div>
                        <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-100"><Info className="h-4 w-4 text-blue-300" /> 라벨링 정보</h4>
                        {/* 탐지된 모든 클래스 표시 */}
                        {selectedImageMetadata.detections.length > 0 && (
                          <div className="mt-2">
                            <p className="text-xs text-slate-400 mb-1">라벨링된 클래스:</p>
                            <div className="flex flex-wrap gap-1">
                              {Array.from(new Set(selectedImageMetadata.detections.map(d => d.class))).map(className => (
                                <Badge key={className} variant="outline" className="border-green-500/40 bg-green-500/10 text-green-200 text-xs">{className}</Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">탐지 개수</span>
                        <span className="font-semibold text-slate-100">{selectedImageMetadata.detections.length}</span>
                      </div>
                      <div className="max-h-64 overflow-y-auto pr-2">
                        <div className="grid grid-cols-2 gap-2">
                          {selectedImageMetadata.detections.map((detection, index) => (
                            <div key={`${selectedImage.id}-det-${index}`} className="rounded-lg border border-slate-800/70 bg-slate-900/80 p-3">
                              <div className="flex items-center gap-2 mb-2">
                                <Badge variant="outline" className="border-blue-500/40 text-blue-200 text-xs">{detection.class}</Badge>
                                {!detection.isGroundTruth && (
                                  <span className="text-xs text-slate-400">{(detection.confidence * 100).toFixed(1)}%</span>
                                )}
                              </div>
                              <div className="text-xs text-slate-500 font-mono">
                                <div>X: {detection.bbox.x1.toFixed(0)} - {detection.bbox.x2.toFixed(0)}</div>
                                <div>Y: {detection.bbox.y1.toFixed(0)} - {detection.bbox.y2.toFixed(0)}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Alert className="border-slate-800/80 bg-slate-900/60 text-slate-300">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>선택한 이미지에 대한 detection 정보를 찾을 수 없습니다.</AlertDescription>
                    </Alert>
                  )}
                </div>
              ) : (
                <div className="flex h-[520px] flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-slate-800/70 bg-slate-950/60 text-center text-slate-500">
                  <ImageIcon className="h-12 w-12" />
                  <div>
                    <p className="text-sm font-medium text-slate-300">이미지를 선택해주세요</p>
                    <p className="mt-1 text-xs text-slate-500">이미지 카드 중 하나를 클릭하면 이곳에 미리보기와 메타데이터가 표시됩니다.</p>
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
        <DialogContent className="max-w-lg border-slate-800/80 text-slate-100">
          <DialogHeader>
            <DialogTitle>데이터셋 등록</DialogTitle>
            <DialogDescription className="text-slate-400">
              YOLO 형식 데이터셋의 경로를 입력하여 데이터베이스에 등록합니다.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <Alert className="bg-blue-900/20 border-blue-500/30">
              <Info className="h-4 w-4 text-blue-400" />
              <AlertDescription className="text-slate-300 text-xs">
                로컬 컴퓨터의 YOLO 데이터셋 폴더를 선택하여 서버로 업로드합니다.<br/>
                폴더 버튼을 클릭하여 이미지 폴더, 라벨 폴더, 클래스 파일을 선택하세요.
              </AlertDescription>
            </Alert>

            <div>
              <Label className="text-slate-200">데이터셋 이름 *</Label>
              <Input
                value={uploadForm.name}
                onChange={(event) =>
                  setUploadForm((prev) => ({ ...prev, name: event.target.value }))
                }
                placeholder="예: COCO_Person_100"
                className="mt-1 border-slate-800 bg-slate-900/70 text-slate-100 placeholder:text-slate-500"
              />
            </div>

            <div>
              <Label className="text-slate-200">이미지 폴더 *</Label>
              <div className="flex gap-2 mt-1">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => imagesFolderInputRef.current?.click()}
                  className="flex-shrink-0 border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800"
                >
                  <FolderOpen className="w-4 h-4 mr-2" />
                  폴더 선택
                </Button>
                <Input
                  value={uploadForm.imageFiles.length > 0 ? `${uploadForm.imageFiles.length}개 파일 선택됨` : ""}
                  readOnly
                  placeholder="이미지 폴더를 선택하세요"
                  className="flex-1 border-slate-800 bg-slate-900/50 text-slate-100 placeholder:text-slate-500 font-mono text-sm"
                />
              </div>
            </div>

            <div>
              <Label className="text-slate-200">라벨 폴더 *</Label>
              <div className="flex gap-2 mt-1">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => labelsFolderInputRef.current?.click()}
                  className="flex-shrink-0 border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800"
                >
                  <FolderOpen className="w-4 h-4 mr-2" />
                  폴더 선택
                </Button>
                <Input
                  value={uploadForm.labelFiles.length > 0 ? `${uploadForm.labelFiles.length}개 파일 선택됨` : ""}
                  readOnly
                  placeholder="라벨 폴더를 선택하세요"
                  className="flex-1 border-slate-800 bg-slate-900/50 text-slate-100 placeholder:text-slate-500 font-mono text-sm"
                />
              </div>
            </div>

            <div>
              <Label className="text-slate-200">클래스 파일 *</Label>
              <div className="flex gap-2 mt-1">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => classesFileInputRef.current?.click()}
                  className="flex-shrink-0 border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  파일 선택
                </Button>
                <Input
                  value={uploadForm.classesFileObject ? uploadForm.classesFileObject.name : ""}
                  readOnly
                  placeholder="classes.txt 파일을 선택하세요"
                  className="flex-1 border-slate-800 bg-slate-900/50 text-slate-100 placeholder:text-slate-500 font-mono text-sm"
                />
              </div>
            </div>

            <div>
              <Label className="text-slate-200">설명 (선택)</Label>
              <Textarea
                value={uploadForm.description}
                onChange={(event) =>
                  setUploadForm((prev) => ({ ...prev, description: event.target.value }))
                }
                rows={2}
                placeholder="데이터셋에 대한 설명을 입력하세요"
                className="mt-1 border-slate-800 bg-slate-900/70 text-slate-100 placeholder:text-slate-500"
              />
            </div>

            {isUploading && (
              <div className="space-y-2">
                <Progress value={uploadProgress} className="h-2" />
                <p className="text-xs text-slate-400">업로드 진행 중... {uploadProgress}%</p>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowUploadDialog(false)} className="border-slate-800 text-slate-300">
              취소
            </Button>
            <Button onClick={handleUploadDataset} disabled={isUploading} className="bg-blue-500 text-white hover:bg-blue-600">
              업로드 시작
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
