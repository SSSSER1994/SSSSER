#include "track_object_distance.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include "boost/format.hpp"


double TrackObjectDistance::welshVarLossFun(double dist, double th, double scale)
{
    double p = 1e-6;
    if (dist < th)
    {
        p = 1 - 1e-6;
    }
    else
    {
        dist -= th;
        dist /= scale;
        p = std::exp(-dist * dist);
    }
    return p;
}

double TrackObjectDistance::scalePositiveProbability(double p, double max_p, double th_p)
{
    if (p <= th_p)
    {
        return p;
    }
    p = (p - th_p) * (max_p - th_p) / (1 - th_p) + th_p;
    return p;
}

double TrackObjectDistance::boundedScalePositiveProbability(double p, double max_p, double min_p)
{
    p = std::max(p, min_p);
    p = (p - min_p) * (max_p - min_p) / (1 - min_p) + min_p;
    return p;
}

float TrackObjectDistance::compute(const Eigen::Vector2f& fused_track,
                                   const Eigen::Vector2f& sensor_object)
{
    float distance = (std::numeric_limits< float >::max)();
    //float min_distance = (std::numeric_limits< float >::max)();

    // double x_diff = std::abs(fused_track(0) - sensor_object(0))/20.0;//
    // double x_similarity = welshVarLossFun(x_diff, 0.5f, 0.3f);
    // x_similarity = scalePositiveProbability(x_similarity, 0.9f, 0.5f);

    // double y_diff = std::max(std::abs(fused_track(1) - sensor_object(1)) - 40.0 * 0.3f, 0.0) / 40.0;//
    // double normalized_y_diff = y_diff * y_diff / 0.2f / 0.2f;
    // double y_similarity = 1 - chiSquaredCdf1TableFun(normalized_y_diff);
    // y_similarity = boundedScalePositiveProbability(y_similarity,0.6f,0.5f);

    // std::vector<double> multiple_similarities = {x_similarity,y_similarity};
    // double fused_similarity = fuseMultipleProbabilities(multiple_similarities);

    // distance = distance_thresh_ * static_cast< float >(1.0 - fused_similarity) /
    //             (1.0f - 0.1f);

    double x_diff = std::abs(fused_track(0) - sensor_object(0));
    double y_diff = std::abs(fused_track(1) - sensor_object(1));
    float min_distance = 0.6*x_diff+0.4*y_diff;
    
    min_distance = std::min(distance, min_distance);
    std::cout<<min_distance<<std::endl;
    return min_distance;
}

double TrackObjectDistance::fuseMultipleProbabilities(const std::vector< double >& probs)
{
    std::vector< double > log_odd_probs = probs;
    auto prob_to_log_odd = [](double p) {
        p = std::max(std::min(p, 1 - 1e-6), 1e-6);
        return std::log(p / (1 - p));
    };
    auto log_odd_to_prob = [](double log_odd_p) {
        double tmp = std::exp(log_odd_p);
        return tmp / (tmp + 1);
    };
    for (auto& log_odd_prob : log_odd_probs)
    {
        log_odd_prob = prob_to_log_odd(log_odd_prob);
    }
    double log_odd_probs_sum = std::accumulate(log_odd_probs.begin(), log_odd_probs.end(), 0.0);
    return log_odd_to_prob(log_odd_probs_sum);
}