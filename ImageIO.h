#pragma once

#include "ImageBuffer_Native.h"

#include "itkImage.h"
#include "itkImportImageFilter.h"

template<class TPixelType>
class ImageIO
{
private:
	typedef itk::Image<TPixelType, 2> ImageType;
	typedef itk::ImportImageFilter<TPixelType, 2> ImportFilterType;

	static typename ImportFilterType::Pointer GetImportFilter(ImageBuffer buffer)
	{
		ImageType::SizeType size = { (unsigned int)buffer.Size.Width, (unsigned int)buffer.Size.Height };
		ImageType::IndexType start = {0, 0};

		if(buffer.Pitch != buffer.Size.Width * PixelFormatUtils::PixelFormatBytesPerPixel(buffer.Format))
			throw std::exception("Buffer must be with Pitch = Width * PixelSize");

		auto importer = ImportFilterType::New();
		importer->SetRegion(ImageType::RegionType(start, size));

		const size_t totalNumberOfPixels = (size_t)size[0] * (size_t)size[1];
		importer->SetImportPointer(buffer.GetLine<TPixelType>(0), totalNumberOfPixels, false);
		return importer;		
	}
public:	

	static typename ImageType::Pointer Import(ImageBuffer buffer)
	{
		auto importer = GetImportFilter(buffer);
		importer->Update();
		return importer->GetOutput();
	}

	static typename ImageType::Pointer Import(MetricImage image)
	{		
		auto importer = GetImportFilter(image.buffer);
		
		ImageType::SpacingType spacing;
		spacing.SetElement(0, image.scaleX);
		spacing.SetElement(1, image.scaleY);
		importer->SetSpacing(spacing);

		importer->Update();
		return importer->GetOutput();
	}

	static void MapToImage(typename ImageType::Pointer image, ImageBuffer buffer)
	{
		const size_t totalNumberOfPixels = (size_t)buffer.Size.Width * (size_t)buffer.Size.Height;

		if(buffer.Pitch != buffer.Size.Width * PixelFormatUtils::PixelFormatBytesPerPixel(buffer.Format))
			throw std::exception("Buffer must be with Pitch = Width * PixelSize");

		ImageType::SizeType size = {(unsigned int)buffer.Size.Width, (unsigned int)buffer.Size.Height};
		ImageType::IndexType start = {0, 0};

		image->SetRegions(ImageType::RegionType(start, size));
		image->GetPixelContainer()->SetImportPointer(buffer.GetLine<TPixelType>(0), totalNumberOfPixels, false);
		image->Allocate();
	}
};