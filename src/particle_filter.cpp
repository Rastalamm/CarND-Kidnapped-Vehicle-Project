/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles  = 75;

    // Lesson 14.5

    default_random_engine gen;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    // This line creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; i++) {
        Particle sample_particle;

        sample_particle.id = i;
        sample_particle.x = dist_x(gen);
        sample_particle.y = dist_y(gen);
        sample_particle.theta = dist_theta(gen);

        particles.push_back(sample_particle);
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Lesson 14.6 - 14.8
    default_random_engine gen;

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    // This line creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    if (yaw_rate < 0.0001) {
        yaw_rate = 0.0001;
    }

    // Update values of x, y, and theta using equations from lesson
    for (int i = 0; i < num_particles; i++) {

        particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen);

        particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(gen);

        particles[i].theta = particles[i].theta + yaw_rate * delta_t + dist_y(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    //for each particle
    for (int i = 0; i < num_particles; i++) {

        double weight = 1;

        // Transform the observations to map space to match the particle spave
        // http://planning.cs.uiuc.edu/node99.html
        // Lesson 14.15
        // for each observation
        for (int j = 0; j < observations.size(); j++) {

            double transformed_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
            double transformed_y = observations[j].y * cos(particles[i].theta) + observations[j].x * sin(particles[i].theta) + particles[i].y;

            Map::single_landmark_s nearest_landmark;
            double min_sensor_distance = sensor_range;

            //calculating the distance between landmarks and transformed observations
            // for each landmark
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {

                Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];
                double distance = fabs(transformed_x - current_landmark.x_f) + fabs(transformed_y - current_landmark.y_f);

                // looking at neatest landmark which matches with the observations
                if (distance < min_sensor_distance) {
                    min_sensor_distance = distance;
                    nearest_landmark = current_landmark;
                }
            }

            // Calculate weight using Normal Distribution
            // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            long double prob = exp(-0.5 *
                (
                    (
                        (nearest_landmark.x_f - transformed_x) * (nearest_landmark.x_f - transformed_x)
                    )
                / (std_landmark[0] * std_landmark[0]) +
                    (
                        (nearest_landmark.y_f - transformed_y) * (nearest_landmark.y_f - transformed_y)
                    )
                / (std_landmark[1] * std_landmark[1]))
                );

            long double norm_const = 2 * M_PI * std_landmark[0] * std_landmark[1];
            weight *= prob / norm_const;
        }

        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    vector<Particle> new_particles;
    std::discrete_distribution<int> discrete_d(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {

        int num = discrete_d(gen);
        new_particles.push_back(particles[num]);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
