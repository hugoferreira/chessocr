package eu.shiftforward.chessocr

import javax.imageio.ImageIO
import java.io.File
import java.awt.image.{DataBufferByte, RescaleOp, BufferedImage}
import java.awt.RenderingHints
import scala.collection.JavaConversions._

object ImageManipulation {
  def processImage(fileName: String)(implicit imageSize: Int, pathPrefix: String = ""): Array[Byte] = {
    val originalImage = ImageIO.read(new File(pathPrefix + fileName))
    val resizedImage = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_BYTE_GRAY)
    val g = resizedImage.createGraphics()
    g.setRenderingHints(Map(RenderingHints.KEY_RENDERING -> RenderingHints.VALUE_RENDER_QUALITY))
    g.setRenderingHints(Map(RenderingHints.KEY_INTERPOLATION -> RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    g.setRenderingHints(Map(RenderingHints.KEY_ANTIALIASING -> RenderingHints.VALUE_ANTIALIAS_ON))
    g.drawImage(originalImage, 0, 0, imageSize, imageSize, null)
    g.dispose()

    val rescaleOp = new RescaleOp(2f, 1, null)
    rescaleOp.filter(resizedImage, resizedImage)

    ImageIO.write(resizedImage, "jpg", new File(pathPrefix + "new" + fileName))
    resizedImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
  }
}
