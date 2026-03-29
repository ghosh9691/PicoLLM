using FluentAssertions;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class LearningRateScheduleTests
{
    [Fact]
    public void Warmup_AtHalfWarmup_ReturnsHalfMaxLr()
    {
        var sched = new LearningRateSchedule(warmupSteps: 100, totalSteps: 1000, maxLr: 1e-4f);
        sched.GetLR(50).Should().BeApproximately(5e-5f, 1e-7f);
    }

    [Fact]
    public void Warmup_AtWarmupEnd_ReturnsMaxLr()
    {
        var sched = new LearningRateSchedule(warmupSteps: 100, totalSteps: 1000, maxLr: 1e-4f);
        sched.GetLR(100).Should().BeApproximately(1e-4f, 1e-7f);
    }

    [Fact]
    public void Decay_AtTotalSteps_ReturnsMinLr()
    {
        var sched = new LearningRateSchedule(warmupSteps: 100, totalSteps: 1000, maxLr: 1e-4f, minLr: 1e-6f);
        sched.GetLR(1000).Should().BeApproximately(1e-6f, 1e-8f);
    }

    [Fact]
    public void Decay_AtMidpoint_IsBetweenMinAndMax()
    {
        var sched = new LearningRateSchedule(warmupSteps: 0, totalSteps: 1000, maxLr: 1e-4f, minLr: 1e-6f);
        float lr = sched.GetLR(500);
        lr.Should().BeGreaterThan(1e-6f).And.BeLessThan(1e-4f);
    }

    [Fact]
    public void NoWarmup_Step0_ReturnsMaxLr()
    {
        var sched = new LearningRateSchedule(warmupSteps: 0, totalSteps: 100, maxLr: 0.01f, minLr: 0f);
        sched.GetLR(0).Should().BeApproximately(0.01f, 1e-6f);
    }

    [Fact]
    public void Schedule_IsMonotonicallyDecreasingAfterWarmup()
    {
        var sched = new LearningRateSchedule(warmupSteps: 10, totalSteps: 100, maxLr: 0.01f, minLr: 0f);
        float prev = sched.GetLR(10);
        for (int s = 11; s <= 100; s++)
        {
            float cur = sched.GetLR(s);
            cur.Should().BeLessThanOrEqualTo(prev + 1e-7f);
            prev = cur;
        }
    }
}
