//#include <filesystem>
#include <stdexcept>
#include <limits>

#include "itkImageRegistrationMethodv4.h"
#include "itkImageRegistrationMethod.h"

#include "itkTranslationTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkEuler2DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkAffineTransform.h"

#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkMutualInformationHistogramImageToImageMetric.h"

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkGradientDescentOptimizer.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"

#include <itkImageMaskSpatialObject.h>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <itkBinaryDilateImageFilter.h>
#include "itkMaskImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImportImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "ImageBuffer_Native.h"
#include "ImageIO.h"
#include "itkKernelImageFilter.h"
#include "itkFlatStructuringElement.h"


#include "itkCommand.h"

constexpr unsigned int Dimension = 2;
using BinaryImageType = itk::Image<unsigned char, Dimension>;
using ShortPixelType = short;
using Short2ImageType = itk::Image<ShortPixelType, Dimension>;
using DoublePixelType = double;
using Double2ImageType = itk::Image<DoublePixelType, 2>;
using TransformType = itk::Euler2DTransform<double>;
using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
using MetricType =
itk::MeanSquaresImageToImageMetricv4<Double2ImageType, Double2ImageType>;
using RegistrationType = itk::
ImageRegistrationMethodv4<Double2ImageType, Double2ImageType, TransformType>;
using MaskType = itk::ImageMaskSpatialObject<Dimension>;
using CompositeTransformType = itk::CompositeTransform<double, Dimension>;

using TransformInitializerType =
itk::CenteredTransformInitializer<TransformType,
    Double2ImageType,
    Double2ImageType>;
//using myCommandType = RegistrationInterfaceCommand<RegistrationType>;
using ResampleFilterType =
itk::ResampleImageFilter<Double2ImageType, Double2ImageType>;
using OutputPixelType = short;
using OutputImageType = itk::Image<OutputPixelType, Dimension>;
using CastFilterType =
itk::CastImageFilter<Double2ImageType, OutputImageType>;
using WriterType = itk::ImageFileWriter<BinaryImageType>;

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
    using Self = RegistrationInterfaceCommand;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    RegistrationInterfaceCommand() = default;

public:
    using RegistrationType = TRegistration;
    using RegistrationPointer = RegistrationType*;
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using OptimizerPointer = OptimizerType*;

    void
        Execute(const itk::Object* object, const itk::EventObject& event) override
    {
        if (!(itk::MultiResolutionIterationEvent().CheckEvent(&event)))
        {
            return;
        }

        std::cout << "\nObserving from class " << object->GetNameOfClass();
        if (!object->GetObjectName().empty())
        {
            std::cout << " \"" << object->GetObjectName() << "\"" << std::endl;
        }

        const auto* registration = static_cast<const RegistrationType*>(object);

        unsigned int currentLevel = registration->GetCurrentLevel();
        typename RegistrationType::ShrinkFactorsPerDimensionContainerType
            shrinkFactors =
            registration->GetShrinkFactorsPerDimension(currentLevel);
        typename RegistrationType::SmoothingSigmasArrayType smoothingSigmas =
            registration->GetSmoothingSigmasPerLevel();

        std::cout << "-------------------------------------" << std::endl;
        std::cout << " Current multi-resolution level = " << currentLevel
            << std::endl;
        std::cout << "    shrink factor = " << shrinkFactors << std::endl;
        std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel]
            << std::endl;
        std::cout << std::endl;
    }

    void
        Execute(itk::Object* caller, const itk::EventObject& event) override
    {
        Execute((const itk::Object*)caller, event);
    }

};

class CommandIterationUpdate : public itk::Command
{
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using OptimizerPointer = const OptimizerType*;

    void
        Execute(itk::Object* caller, const itk::EventObject& event) override
    {
        Execute((const itk::Object*)caller, event);
    }

    void
        Execute(const itk::Object* object, const itk::EventObject& event) override
    {
        auto optimizer = static_cast<OptimizerPointer>(object);
        if (!(itk::IterationEvent().CheckEvent(&event)))
        {
            return;
        }
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << optimizer->GetCurrentPosition() << "   ";
        std::cout << m_CumulativeIterationIndex++ << std::endl;
    }

private:
    unsigned int m_CumulativeIterationIndex{ 0 };
};


BinaryImageType::Pointer DilateImage(BinaryImageType::Pointer sourceImage, int ballRadius)
{
    using StructuringElementType = itk::FlatStructuringElement< Dimension >;
    StructuringElementType::RadiusType radius;
    radius.Fill(ballRadius);
    StructuringElementType structuringElement = StructuringElementType::Ball(radius);

    using BinaryDilateImageFilterType = itk::BinaryDilateImageFilter<BinaryImageType, BinaryImageType, StructuringElementType>;

    BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
    dilateFilter->SetInput(sourceImage);
    dilateFilter->SetKernel(structuringElement);
    dilateFilter->SetForegroundValue(255); 
    dilateFilter->Update();

    return dilateFilter->GetOutput();
}

Double2ImageType::Pointer CastImageShortDouble(Short2ImageType::Pointer source)
{
    using CastFilter = itk::CastImageFilter<Short2ImageType, Double2ImageType>;
    auto caster = CastFilter::New();
    caster->SetInput(source);
    caster->Update();
    return caster->GetOutput();
}

Short2ImageType::Pointer GetITKImageFromBuffer(ImageBuffer buffer, float widthScale, float heightScale)
{
    float correctSpacing[Dimension] = { widthScale, heightScale };
    ImageIO<ShortPixelType> importerImage;

    auto itkImage = importerImage.Import(buffer);
    itkImage->SetSpacing(correctSpacing);
    return itkImage;
}

BinaryImageType::Pointer GetMask(itk::Image<short, 2>::Pointer source, short threshold)
{
    BinaryImageType::Pointer mask = BinaryImageType::New();
    mask->SetRegions(source->GetLargestPossibleRegion());
    auto width = source->GetLargestPossibleRegion().GetSize()[0];
    auto height = source->GetLargestPossibleRegion().GetSize()[1];
    mask->Allocate();

    auto maskPointer = mask->GetBufferPointer();
    auto sourcePointer = source->GetBufferPointer();

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            if (*(sourcePointer + i + j * width) > threshold)
                *(maskPointer + i + j * width) = 255;
            else
                *(maskPointer + i + j * width) = 0;
        }
    }

    return mask;
}

void CheckPitchCorrection(ImageBuffer buffer)
{
    const size_t totalNumberOfPixels = (size_t)buffer.Size.Width * (size_t)buffer.Size.Height;
    if (buffer.Pitch != buffer.Size.Width * PixelFormatUtils::PixelFormatBytesPerPixel(buffer.Format))
        throw std::exception("Buffer must be with Pitch = Width * PixelSize");
}

OptimizerType::Pointer GetOptimizer()
{
    OptimizerType::Pointer optimizer = OptimizerType::New();
    
    optimizer->SetLearningRate(0.1);
    optimizer->SetNumberOfIterations(500);
    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetMinimumStepLength(0.0001);
    
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    return optimizer;
}

MetricType::Pointer GetMetric(BinaryImageType::Pointer maskFixed, BinaryImageType::Pointer maskMoving)
{
    MetricType::Pointer       metric = MetricType::New();

    MaskType::Pointer spatialObjectMaskFixed = MaskType::New();
    spatialObjectMaskFixed->SetImage(maskFixed);
    spatialObjectMaskFixed->Update();
    metric->SetFixedImageMask(spatialObjectMaskFixed);
    
    MaskType::Pointer spatialObjectMaskMoving = MaskType::New();
    spatialObjectMaskMoving->SetImage(maskMoving);
    spatialObjectMaskMoving->Update();
    metric->SetMovingImageMask(spatialObjectMaskMoving);

    return metric;
}

TransformType::Pointer GetCenteredTransform(Double2ImageType::Pointer fixed, Double2ImageType::Pointer moving)
{
    TransformType::Pointer    transform = TransformType::New();
    TransformInitializerType::Pointer initializer =
        TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixed);
    initializer->SetMovingImage(moving);
    initializer->MomentsOn();
    initializer->InitializeTransform();

    return transform;
}

RegistrationType::Pointer GetRegistration(OptimizerType::Pointer optimizer, MetricType::Pointer metric,
    Double2ImageType::Pointer fixedDouble, Double2ImageType::Pointer movingDouble,
    TransformType::Pointer transform)
{
    RegistrationType::Pointer registration = RegistrationType::New();

    registration->SetOptimizer(optimizer);
    registration->SetMetric(metric);

    registration->SetFixedImage(fixedDouble);
    registration->SetMovingImage(movingDouble);

    constexpr unsigned int numberOfLevels = 4;
    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(numberOfLevels);
    shrinkFactorsPerLevel[0] = 8;
    shrinkFactorsPerLevel[1] = 4;
    shrinkFactorsPerLevel[2] = 2;
    shrinkFactorsPerLevel[3] = 1;

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(numberOfLevels);
    smoothingSigmasPerLevel[0] = 0;
    smoothingSigmasPerLevel[1] = 0;
    smoothingSigmasPerLevel[2] = 0;
    smoothingSigmasPerLevel[3] = 0;

    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);

    //myCommandType::Pointer command = myCommandType::New();
    RegistrationInterfaceCommand<RegistrationType>::Pointer command =
        RegistrationInterfaceCommand<RegistrationType>::New();
    registration->AddObserver(itk::MultiResolutionIterationEvent(), command);

    return registration;
}

Double2ImageType::Pointer ResampleImage(Double2ImageType::Pointer movingDouble, TransformType::Pointer transform)
{
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetTransform(transform);
    resampler->SetInput(movingDouble);

    resampler->SetSize(movingDouble->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(movingDouble->GetOrigin());
    resampler->SetOutputSpacing(movingDouble->GetSpacing());
    resampler->SetOutputDirection(movingDouble->GetDirection());

    return resampler->GetOutput();
}

Short2ImageType::Pointer CastImageDoubleShort(Double2ImageType::Pointer source)
{
    using CastFilter = itk::CastImageFilter<Double2ImageType, Short2ImageType>;
    auto caster = CastFilter::New();
    caster->SetInput(source);
    caster->Update();
    return caster->GetOutput();
}

void CopyFromImageToBuffer(Short2ImageType::Pointer castedImage, ImageBuffer correctedBuffer)
{
    auto pointerToCastedImage = castedImage->GetBufferPointer();

    for (int i = 0; i < correctedBuffer.Size.Width; ++i)
        for (int j = 0; j < correctedBuffer.Size.Height; ++j)
            correctedBuffer.Set(i, j, *(pointerToCastedImage + i + j * correctedBuffer.Size.Width));
}

extern "C" __declspec(dllexport) int itkMotionCorrection(ImageBuffer fixedBuffer, ImageBuffer movingBuffer, ImageBuffer correctedBuffer,
    float widthScale, float heightScale, short threshold)
{   
    CheckPitchCorrection(fixedBuffer);
    CheckPitchCorrection(movingBuffer);

    auto fixedImage = GetITKImageFromBuffer(fixedBuffer, widthScale, heightScale);
    auto movingImage = GetITKImageFromBuffer(movingBuffer, widthScale, heightScale);

    auto maskFixed = GetMask(fixedImage, threshold);
    auto maskMoving = GetMask(movingImage, threshold);

    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(maskFixed);
    writer->SetFileName("maskFixed.png");
    writer->Update();
    writer->SetInput(maskMoving);
    writer->SetFileName("maskMoving.png");
    writer->Update();

    auto fixedDouble = CastImageShortDouble(fixedImage);
    auto movingDouble = CastImageShortDouble(movingImage);
    
    
    auto optimizer = GetOptimizer();
    auto metric = GetMetric(maskFixed, maskMoving);
    auto transform = GetCenteredTransform(fixedDouble, movingDouble);
    
    auto registration = GetRegistration(optimizer,
                                        metric,
                                        fixedDouble,
                                        movingDouble,
                                        transform);
    try
    {
        registration->Update();
        std::cout << "Optimizer stop condition: "
            << registration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
    }
    catch (const itk::ExceptionObject& err)
    {
        std::cout << "ExceptionObject caught !" << std::endl;
        std::cout << err << std::endl;
        return EXIT_FAILURE;
    }

    auto resampledImage = ResampleImage(movingDouble, registration->GetModifiableTransform());
    auto castedImage = CastImageDoubleShort(resampledImage);
    
    CopyFromImageToBuffer(castedImage, correctedBuffer);

    return 0;
}

