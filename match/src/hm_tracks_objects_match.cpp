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
#include "hm_tracks_objects_match.h"

bool HMTrackersObjectsAssociation::associate(std::vector<Eigen::Vector2f>& measurement,
                                             std::vector<Eigen::Vector2f>& tracker, AssociationResult& associationresult)
{
    std::vector< std::vector< double > > association_mat;
    // if (tracker.empty() || measurement.empty())
    // {
        associationresult.unassigned_tracks.resize(tracker.size());
        associationresult.unassigned_measurements.resize(measurement.size());
        std::iota(associationresult.unassigned_tracks.begin(), associationresult.unassigned_tracks.end(), 0);
        std::iota(
            associationresult.unassigned_measurements.begin(), associationresult.unassigned_measurements.end(), 0);
        // return true;
    // }

    // int measurement_sensor_id = measurement[0]->getSensorId();
    // double measurement_timestamp = sensor_objects[0]->getTimestamp();
    computeAssociationDistanceMat(tracker,
                                  measurement,
                                  associationresult.unassigned_tracks,
                                  associationresult.unassigned_measurements,
                                  &association_mat);
    //std::cout<<association_mat.size()<<std::endl;
    int num_track = static_cast< int >(tracker.size());
    int num_measurement = static_cast< int >(measurement.size());
    associationresult.track2measurements_dist.assign(num_track, 0);
    associationresult.measurement2track_dist.assign(num_measurement, 0);
    std::vector< int > track_ind_g2l;
    track_ind_g2l.resize(num_track, -1);
    for (size_t i = 0; i < associationresult.unassigned_tracks.size(); i++)
    {
        track_ind_g2l[associationresult.unassigned_tracks[i]] = static_cast< int >(i);
    }
    std::vector< int > measurement_ind_g2l;
    measurement_ind_g2l.resize(num_measurement, -1);
    std::vector< size_t > measurement_ind_l2g = associationresult.unassigned_measurements;

    for (size_t i = 0; i < associationresult.unassigned_measurements.size(); i++)
    {
        measurement_ind_g2l[associationresult.unassigned_measurements[i]] = static_cast< int >(i);
    }

    std::vector< size_t > track_ind_l2g = associationresult.unassigned_tracks;

    if (associationresult.unassigned_tracks.empty() || associationresult.unassigned_measurements.empty())
    {
        return true;
    }
    bool state = minimizeAssignment(association_mat,
                                    track_ind_l2g,
                                    measurement_ind_l2g,
                                    &associationresult.assignments,
                                    &associationresult.unassigned_tracks,
                                    &associationresult.unassigned_measurements);
    return state;
}

bool HMTrackersObjectsAssociation::minimizeAssignment(const std::vector< std::vector< double > >& association_mat,
                                                      const std::vector< size_t >& track_ind_l2g,
                                                      const std::vector< size_t >& measurement_ind_l2g,
                                                      std::vector< TrackMeasurmentPair >* assignments,
                                                      std::vector< size_t >* unassigned_tracks,
                                                      std::vector< size_t >* unassigned_measurements)
{

    GatedHungarianMatcher< float >::OptimizeFlag opt_flag = GatedHungarianMatcher< float >::OptimizeFlag::kOptMin;
    SecureMat< float >* global_costs = optimizer_.mutable_global_costs();
    int rows = static_cast< int >(unassigned_tracks->size());
    int cols = static_cast< int >(unassigned_measurements->size());

    global_costs->resize(rows, cols);
    for (int r_i = 0; r_i < rows; r_i++)
    {
        for (int c_i = 0; c_i < cols; c_i++)
        {
            (*global_costs)(r_i, c_i) = static_cast< float >(association_mat[r_i][c_i]);
        }
    }
    std::vector< TrackMeasurmentPair > local_assignments;
    std::vector< size_t > local_unassigned_tracks;
    std::vector< size_t > local_unassigned_measurements;
    optimizer_.match(static_cast< float >(s_match_distance_thresh_),
                     static_cast< float >(s_match_distance_bound_),
                     opt_flag,
                     &local_assignments,
                     &local_unassigned_tracks,
                     &local_unassigned_measurements);
    for (auto assign : local_assignments)
    {
        assignments->push_back(std::make_pair(track_ind_l2g[assign.first], measurement_ind_l2g[assign.second]));
    }
    unassigned_tracks->clear();
    unassigned_measurements->clear();
    for (auto un_track : local_unassigned_tracks)
    {
        unassigned_tracks->push_back(track_ind_l2g[un_track]);
    }
    for (auto un_mea : local_unassigned_measurements)
    {
        unassigned_measurements->push_back(measurement_ind_l2g[un_mea]);
    }
    return true;
}

void HMTrackersObjectsAssociation::computeAssociationDistanceMat(const std::vector< Eigen::Vector2f >& fusion_tracks,
                                       const std::vector< Eigen::Vector2f >& sensor_objects,
                                       const std::vector< size_t >& unassigned_tracks,
                                       const std::vector< size_t >& unassigned_measurements,
                                       std::vector< std::vector< double > >* association_mat)
{
    association_mat->resize(unassigned_tracks.size());
    for (size_t i = 0; i < unassigned_tracks.size(); ++i)
    {
        int fusion_idx = static_cast< int >(unassigned_tracks[i]);
        (*association_mat)[i].resize(unassigned_measurements.size());
        const Eigen::Vector2f& fusion_track = fusion_tracks[fusion_idx];
        for (size_t j = 0; j < unassigned_measurements.size(); ++j)
        {
            int sensor_idx = static_cast< int >(unassigned_measurements[j]);
            const Eigen::Vector2f& sensor_object = sensor_objects[sensor_idx];
            double distance = 4.0;
            // to do
            double center_dst =
                (sensor_object- fusion_track).norm();
            if (center_dst < s_association_center_dist_threshold_)
            {
                distance = track_object_distance_.compute(fusion_track, sensor_object);
                //std::cout<<distance<<std::endl;
            }
            (*association_mat)[i][j] = distance;
        }
    }
}
