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
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"

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

#include <itkOptimizerParameterScalesEstimator.h>

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
itk::CorrelationImageToImageMetricv4<Double2ImageType, Double2ImageType>;
using RegistrationType = itk::
ImageRegistrationMethodv4<Double2ImageType, Double2ImageType, TransformType>;
using MaskType = itk::ImageMaskSpatialObject<Dimension>;
using CompositeTransformType = itk::CompositeTransform<double, Dimension>;
using TransformInitializerType =
itk::CenteredTransformInitializer<TransformType,
    Double2ImageType,
    Double2ImageType>;
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
using myCommandType = RegistrationInterfaceCommand<RegistrationType>;

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
        previous = optimizer->GetValue();
    }

private:
    unsigned int m_CumulativeIterationIndex{ 0 };
    double previous = 0.;
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

BinaryImageType::Pointer GetMask(Short2ImageType::Pointer source, short threshold)
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

OptimizerType::Pointer GetOptimizer(MetricType::Pointer metric)
{
    OptimizerType::Pointer optimizer = OptimizerType::New();
    using ScalesEstimatorType =
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(true);

    optimizer->SetScalesEstimator(scalesEstimator);
    optimizer->SetMetric(metric);
    optimizer->SetLearningRate(0.01);
    
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

    bool isMetricMutualInformation = false;
    if (isMetricMutualInformation)
    {
        metric->SetUseMovingImageGradientFilter(false);
        metric->SetUseFixedImageGradientFilter(false);
    }

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
    registration->SetInitialTransform(transform);
    registration->InPlaceOn();
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

    myCommandType::Pointer stageObserver = myCommandType::New();
    registration->AddObserver(itk::MultiResolutionIterationEvent(), stageObserver);

    return registration;
}

Double2ImageType::Pointer ResampleImage(Double2ImageType::Pointer ImageDouble, TransformType::Pointer transform)
{
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetTransform(transform);
    resampler->SetInput(ImageDouble);

    resampler->SetSize(ImageDouble->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(ImageDouble->GetOrigin());
    resampler->SetOutputSpacing(ImageDouble->GetSpacing());
    resampler->SetOutputDirection(ImageDouble->GetDirection());
    resampler->Update();

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

void CopyFromImageToBuffer(Short2ImageType::Pointer image, ImageBuffer buffer)
{
    auto pointerToCastedImage = image->GetBufferPointer();
    auto size = image->GetLargestPossibleRegion().GetSize();
    auto width = size[0];
    auto height = size[1];

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            buffer.Set(i, j, *(pointerToCastedImage + i + j * width));
}

extern "C" __declspec(dllexport) int itkMotionCorrection(ImageBuffer fixedRescaledBuffer, ImageBuffer movingRescaledBuffer, 
                                                         ImageBuffer movingSourceBuffer, ImageBuffer correctedBuffer,
                                                        float widthScale, float heightScale, short threshold)
{   
    CheckPitchCorrection(fixedRescaledBuffer);
    CheckPitchCorrection(movingRescaledBuffer);

    auto fixedImage = GetITKImageFromBuffer(fixedRescaledBuffer, widthScale, heightScale);
    auto movingImage = GetITKImageFromBuffer(movingRescaledBuffer, widthScale, heightScale);

    auto maskFixed = GetMask(fixedImage, threshold);
    auto maskMoving = GetMask(movingImage, threshold);

    auto dilatedMaskFixed = DilateImage(maskFixed, 5);
    auto dilatedMaskMoving = DilateImage(maskMoving, 5);

    auto fixedDouble = CastImageShortDouble(fixedImage);
    auto movingDouble = CastImageShortDouble(movingImage);
    
    auto metric = GetMetric(dilatedMaskFixed, dilatedMaskMoving);
    auto optimizer = GetOptimizer(metric);
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
    
    auto movingSource = GetITKImageFromBuffer(movingSourceBuffer, widthScale, heightScale);
    auto movingSourceDouble = CastImageShortDouble(movingSource);
    //auto resampledImage = ResampleImage(movingDouble, registration->GetModifiableTransform());
    auto resampledImage = ResampleImage(movingSourceDouble, registration->GetModifiableTransform());
    auto castedImage = CastImageDoubleShort(resampledImage);
    
    CopyFromImageToBuffer(castedImage, correctedBuffer);

    return 0;
}

