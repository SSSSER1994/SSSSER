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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gated_hungarian_bigraph_matcher.h"
#include "track_object_distance.h"


typedef std::pair< size_t, size_t > TrackMeasurmentPair;

struct AssociationResult
{
    std::vector< TrackMeasurmentPair > assignments;
    std::vector< size_t > unassigned_tracks;
    std::vector< size_t > unassigned_measurements;
    std::vector< double > track2measurements_dist;
    std::vector< double > measurement2track_dist;
};

/**
* @brief HMTrackersObjectsAssociation class that match the sensor objects with fusion tracks
* Mainly output the match result of sensor measurements and last fusion tracks
*/
class HMTrackersObjectsAssociation
{
  public:
    HMTrackersObjectsAssociation() = default;
    ~HMTrackersObjectsAssociation() = default;
    /**
     * @brief set the match distance thresh for sensor measurement and fusion tracks
     */
    bool init()
    {
        return true;
    }
    /**
     * @brief Calculate the distance between sensor measurements and fusion tracks
     * @param sensor_measurements sensor measurements input
     * @param scene fusion tracks
     * @param association_result associate result for sensor_measurements and tracks
     */
    bool associate(std::vector<Eigen::Vector2f>& measurement, std::vector<Eigen::Vector2f>& tracker, AssociationResult& associationresult);

  private:
    void computeAssociationDistanceMat(const std::vector< Eigen::Vector2f >& fusion_tracks,
                                       const std::vector< Eigen::Vector2f >& sensor_objects,
                                       const std::vector< size_t >& unassigned_tracks,
                                       const std::vector< size_t >& unassigned_measurements,
                                       std::vector< std::vector< double > >* association_mat);

    bool minimizeAssignment(const std::vector< std::vector< double > >& association_mat,
                            const std::vector< size_t >& track_ind_l2g,
                            const std::vector< size_t >& measurement_ind_l2g,
                            std::vector< TrackMeasurmentPair >* assignments,
                            std::vector< size_t >* unassigned_tracks,
                            std::vector< size_t >* unassigned_measurements);

    void computeDistance(const std::vector< Eigen::Vector2f >& fusion_tracks,
                         const std::vector< Eigen::Vector2f >& sensor_objects,
                         const std::vector< size_t >& unassigned_fusion_track,
                         const std::vector< int >& track_ind_g2l,
                         const std::vector< int >& measurement_ind_g2l,
                         const std::vector< size_t >& measurement_ind_l2g,
                         const std::vector< std::vector< double > >& association_mat,
                         AssociationResult* association_result);

    void generateUnassignedData(size_t track_num,
                                size_t objects_num,
                                const std::vector< TrackMeasurmentPair >& assignments,
                                std::vector< size_t >* unassigned_tracks,
                                std::vector< size_t >* unassigned_objects);

    GatedHungarianMatcher< float > optimizer_;
    TrackObjectDistance track_object_distance_;
    double s_match_distance_thresh_ = 15 ;
    double s_match_distance_bound_ = 40;
    double s_association_center_dist_threshold_ = 150;
};

