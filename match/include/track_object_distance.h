/*
 * Copyright (C) SAIC VOLKSWAGEN Automotive Co., Ltd(SVW). All Rights Reserved.
 * This software is the confidential and proprietary information of SVW.
 * You shall not disclose such Confidential Information and shall use it only in
 * accordance with the terms of the license agreement you entered into with SVW.
 * Some implementations are refer to open source projects with permissive
 * open source licenses.
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PsvwOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

#include <string>
#include <vector>
#include "Eigen/Core"
#include "Eigen/StdVector"
#include "Eigen/StdVector"
#include "chi_squared_1_distribution.h"

/**
* @brief In HMTrackersObjectsAssociation class, we mainly output the associate results,
* And in this TrackObjectDistance class, we mainly calculate the associate Matrix, and then
* Use the associate Matrix to calculate the associate results
*/
class TrackObjectDistance
{
  public:
    TrackObjectDistance() = default;
    ~TrackObjectDistance() = default;
    /**
     * @brief to do compute the distance between input fused track and sensor object
     * @params[IN] fused_track: each fused track
     * @params[IN] sensor_object: sensor measurement
     * @params[IN] options: options of track object distanace computation
     * @return returns the distance between  fused track and sensor object
     */
    float compute(const Eigen::Vector2f& fused_track, const Eigen::Vector2f& sensor_object);

  private:

    double welshVarLossFun(double dist, double th, double scale);
    double scalePositiveProbability(double p, double max_p, double th_p);
    double boundedScalePositiveProbability(double p, double max_p, double min_p);
    double fuseMultipleProbabilities(const std::vector< double >& probs);
    float distance_thresh_ = 8.0f;
};

