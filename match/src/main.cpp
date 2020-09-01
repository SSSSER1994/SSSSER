#include <memory>
#include <string>
#include <vector>
#include "Eigen/Core"
#include "hm_tracks_objects_match.h"



int main()
{
    std::vector<Eigen::Vector2f> measurement;
    std::vector<Eigen::Vector2f> tracker;
    Eigen::Vector2f point1(366,224);tracker.push_back(point1);
    Eigen::Vector2f point2(273,175);tracker.push_back(point2);
    Eigen::Vector2f point3(517,188);tracker.push_back(point3);
    Eigen::Vector2f point4(504,225);tracker.push_back(point4);
    Eigen::Vector2f point5(172,235);tracker.push_back(point5);
    Eigen::Vector2f point6(572,245);tracker.push_back(point6);
    Eigen::Vector2f point7(343,224);tracker.push_back(point7);
    Eigen::Vector2f point8(227,178);tracker.push_back(point8);
    Eigen::Vector2f point9(473,187);tracker.push_back(point9);
    Eigen::Vector2f point10(485,227);tracker.push_back(point10);
//8,0;   9,1;   3,2;    2,3;   0,4 ;  1,5
    Eigen::Vector2f pointa(471,191);measurement.push_back(pointa);
    Eigen::Vector2f pointb(480,224);measurement.push_back(pointb);
    Eigen::Vector2f pointc(498,223);measurement.push_back(pointc);
    Eigen::Vector2f pointd(512,193);measurement.push_back(pointd);
    Eigen::Vector2f pointe(364,225);measurement.push_back(pointe);
    Eigen::Vector2f pointf(280,185);measurement.push_back(pointf);

    AssociationResult associationresult;
    HMTrackersObjectsAssociation match;
    match.associate(measurement,tracker,associationresult);
    for (size_t i = 0; i < associationresult.assignments.size(); i++)
    {
        std::cout<<associationresult.assignments[i].first<<" , "<<associationresult.assignments[i].second<<std::endl;
    }
    return 0;
}