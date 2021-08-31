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
#include "itkRescaleIntensityImageFilter.h"

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include <itkGradientDescentOptimizer.h>
#include <itkConjugateGradientLineSearchOptimizerv4.h>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImportImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "ImageBuffer_Native.h"
#include "ImageIO.h"
//#include "C:\Users\owchi\source\repos\dev3\Native\Alda.Native.Run\ImageIO.h"

namespace fs = std::filesystem;

#include "itkCommand.h"
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

extern "C" __declspec(dllexport) int itkMotionCorrection(ImageBuffer fixedBuffer, ImageBuffer movingBuffer, ImageBuffer correctedBuffer,
    float widthScale, float heightScale)
{    
    constexpr unsigned int Dimension = 2;
    using ShortPixelType  = short;
    using Short2ImageType = itk::Image<ShortPixelType, Dimension>;
    
    const size_t totalNumberOfPixels = (size_t)fixedBuffer.Size.Width * (size_t)fixedBuffer.Size.Height;
    if (fixedBuffer.Pitch != fixedBuffer.Size.Width * PixelFormatUtils::PixelFormatBytesPerPixel(fixedBuffer.Format))
        throw std::exception("Buffer must be with Pitch = Width * PixelSize");

    float correctSpacing[Dimension] = { widthScale, heightScale };
    ImageIO<ShortPixelType> importerImage;
    auto fixedImage = importerImage.Import(fixedBuffer);
    fixedImage->SetSpacing(correctSpacing);
    auto movingImage = importerImage.Import(movingBuffer);
    movingImage->SetSpacing(correctSpacing);

    using DoublePixelType = double;
    using Double2ImageType = itk::Image<DoublePixelType, 2>;
    using CastShortDouble = itk::CastImageFilter<Short2ImageType, Double2ImageType>;
    CastShortDouble::Pointer castFixedShortDouble = CastShortDouble::New();
    castFixedShortDouble->SetInput(fixedImage); 
    castFixedShortDouble->Update();
    CastShortDouble::Pointer castMovingShortDouble = CastShortDouble::New();
    castMovingShortDouble->SetInput(movingImage);
    castMovingShortDouble->Update();

    //using TransformType = itk::Euler2DTransform<double>;
    using TransformType = itk::Similarity2DTransform<double>;
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using MetricType =
        itk::MeanSquaresImageToImageMetricv4<Double2ImageType,
        Double2ImageType>;
    using RegistrationType = itk::
        ImageRegistrationMethodv4<Double2ImageType, Double2ImageType, TransformType>;

    TransformType::Pointer    transform = TransformType::New();
    OptimizerType::Pointer    optimizer = OptimizerType::New();
    MetricType::Pointer       metric = MetricType::New();
    RegistrationType::Pointer registration = RegistrationType::New();

    registration->SetOptimizer(optimizer);
    registration->SetMetric(metric);

    registration->SetFixedImage(castFixedShortDouble->GetOutput());
    registration->SetMovingImage(castMovingShortDouble->GetOutput());

    using CompositeTransformType = itk::CompositeTransform<double, Dimension>;
    CompositeTransformType::Pointer compositeTransform =
        CompositeTransformType::New();
    

    using TransformInitializerType =
        itk::CenteredTransformInitializer<TransformType,
        Double2ImageType,
        Double2ImageType>;
    TransformInitializerType::Pointer initializer =
        TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(castFixedShortDouble->GetOutput());
    initializer->SetMovingImage(castMovingShortDouble->GetOutput());
    initializer->MomentsOn();
    initializer->InitializeTransform();

    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    optimizer->SetLearningRate(0.1);
    optimizer->SetNumberOfIterations(200);
    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetMinimumStepLength(0.01);

    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);
    
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
    
    using CommandType = RegistrationInterfaceCommand<RegistrationType>;
    CommandType::Pointer command = CommandType::New();
    registration->AddObserver(itk::MultiResolutionIterationEvent(), command);

    try
    {
        registration->Update();
        compositeTransform->AddTransform(registration->GetModifiableTransform());
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

    using ATransformType = itk::AffineTransform<double, Dimension>;
    using AOptimizerType =
        itk::ConjugateGradientLineSearchOptimizerv4Template<double>;
    using ARegistrationType = itk::ImageRegistrationMethodv4<Double2ImageType,
        Double2ImageType,
        ATransformType>;

    AOptimizerType::Pointer    affineOptimizer = AOptimizerType::New();
    MetricType::Pointer        affineMetric = MetricType::New();
    ARegistrationType::Pointer affineRegistration = ARegistrationType::New();

    affineRegistration->SetOptimizer(affineOptimizer);
    affineRegistration->SetMetric(affineMetric);

    affineRegistration->SetFixedImage(castFixedShortDouble->GetOutput());
    affineRegistration->SetMovingImage(castMovingShortDouble->GetOutput());
    affineRegistration->SetMovingInitialTransform(compositeTransform);
    affineRegistration->SetObjectName("AffineRegistration");

    using FixedImageCalculatorType =
        itk::ImageMomentsCalculator<Double2ImageType>;

    FixedImageCalculatorType::Pointer fixedCalculator =
        FixedImageCalculatorType::New();
    fixedCalculator->SetImage(castFixedShortDouble->GetOutput());
    fixedCalculator->Compute();
    auto fixedCenter = fixedCalculator->GetCenterOfGravity();

    ATransformType::Pointer affineTx = ATransformType::New();
    const unsigned int numberOfFixedParameters =
        affineTx->GetFixedParameters().Size();
    ATransformType::ParametersType fixedParameters(numberOfFixedParameters);
    for (unsigned int i = 0; i < numberOfFixedParameters; ++i)
    {
        fixedParameters[i] = fixedCenter[i];
    }
    affineTx->SetFixedParameters(fixedParameters);

    affineRegistration->SetInitialTransform(affineTx);
    affineRegistration->InPlaceOn();

    using ScalesEstimatorType =
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(affineMetric);
    scalesEstimator->SetTransformForward(true);

    affineOptimizer->SetScalesEstimator(scalesEstimator);
    affineOptimizer->SetDoEstimateLearningRateOnce(true);
    affineOptimizer->SetDoEstimateLearningRateAtEachIteration(false);

    affineOptimizer->SetLowerLimit(0);
    affineOptimizer->SetUpperLimit(2);
    affineOptimizer->SetEpsilon(0.2);
    affineOptimizer->SetNumberOfIterations(200);
    affineOptimizer->SetMinimumConvergenceValue(1e-6);
    affineOptimizer->SetConvergenceWindowSize(5);

    CommandIterationUpdate::Pointer observer2 = CommandIterationUpdate::New();
    affineOptimizer->AddObserver(itk::IterationEvent(), observer2);

    constexpr unsigned int numberOfLevels2 = 2;

    ARegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel2;
    shrinkFactorsPerLevel2.SetSize(numberOfLevels2);
    shrinkFactorsPerLevel2[0] = 2;
    shrinkFactorsPerLevel2[1] = 1;

    ARegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel2;
    smoothingSigmasPerLevel2.SetSize(numberOfLevels2);
    smoothingSigmasPerLevel2[0] = 1;
    smoothingSigmasPerLevel2[1] = 0;

    affineRegistration->SetNumberOfLevels(numberOfLevels2);
    affineRegistration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel2);
    affineRegistration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel2);

    using AffineCommandType = RegistrationInterfaceCommand<ARegistrationType>;
    AffineCommandType::Pointer command2 = AffineCommandType::New();
    affineRegistration->AddObserver(itk::MultiResolutionIterationEvent(),
        command2);

    try
    {
        affineRegistration->Update();
        compositeTransform->AddTransform(affineRegistration->GetModifiableTransform());
        std::cout
            << "Optimizer stop condition: "
            << affineRegistration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
    }
    catch (const itk::ExceptionObject& err)
    {
        std::cout << "ExceptionObject caught !" << std::endl;
        std::cout << err << std::endl;
        return EXIT_FAILURE;
    }


    using ResampleFilterType =
        itk::ResampleImageFilter<Double2ImageType, Double2ImageType>;
    ResampleFilterType::Pointer resample = ResampleFilterType::New();
    resample->SetTransform(compositeTransform);
    resample->SetInput(castMovingShortDouble->GetOutput());

    Double2ImageType::Pointer fixedImageTemp = castFixedShortDouble->GetOutput();
    resample->SetSize(fixedImageTemp->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(fixedImageTemp->GetOrigin());
    resample->SetOutputSpacing(fixedImageTemp->GetSpacing());
    resample->SetOutputDirection(fixedImageTemp->GetDirection());
    
    using OutputPixelType = short;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;
    using CastFilterType =
        itk::CastImageFilter<Double2ImageType, OutputImageType>;
    CastFilterType::Pointer caster = CastFilterType::New();

    caster->SetInput(resample->GetOutput());
    caster->Update();
    auto pointerToCastedImage = caster->GetOutput()->GetBufferPointer();

    for (int i = 0; i < correctedBuffer.Size.Width; ++i)
    {
        for (int j = 0; j < correctedBuffer.Size.Height; ++j)
        {
            correctedBuffer.Set(i, j, *(pointerToCastedImage + i + j * correctedBuffer.Size.Width) );
        }
    }

    return 0;
}

