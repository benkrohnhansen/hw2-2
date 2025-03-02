#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
// Put any static global variables here that you will use throughout the simulation.

// Apply force between two particles
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

double lower_bound;
double upper_bound;

int printingRank = 0; // Set this to control which rank prints debug output

std::vector<particle_t> local_parts; // Store local particles globally for use in simulate_one_step

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_height = size / num_procs;
    lower_bound = rank * row_height;
    upper_bound = (rank + 1) * row_height;
    
    // Print all initial particles
    if (rank == printingRank) {
        std::cout << "[Rank " << rank << "] Initial particles:\n";
        for (int i = 0; i < num_parts; i++) {
            std::cout << "  Particle " << i << ": (x: " << parts[i].x << ", y: " << parts[i].y << ")\n";
        }
    }
    
    // Clear local_parts and assign relevant particles to this rank
    local_parts.clear();
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y >= lower_bound && parts[i].y < upper_bound) {
            local_parts.push_back(parts[i]);
        }
    }
    
    // Print local particles after assignment
    if (rank == printingRank) {
        std::cout << "[Rank " << rank << "] Local particles after assignment:\n";
        for (size_t i = 0; i < local_parts.size(); i++) {
            std::cout << "  Particle " << i << ": (x: " << local_parts[i].x << ", y: " << local_parts[i].y << ")\n";
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Vectors to store particles that need to be sent and received
    std::vector<double> ghost_to_above;
    std::vector<double> ghost_to_below;
    std::vector<double> ghost_from_above;
    std::vector<double> ghost_from_below;

    // Define rank above and below
    int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
    int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

    // Iterate over local particles to determine which need to be sent
    for (size_t i = 0; i < local_parts.size(); i++) {
        if (upper_bound - local_parts[i].y < cutoff) {
            ghost_to_above.push_back(local_parts[i].x);
            ghost_to_above.push_back(local_parts[i].y);
        }
        if (local_parts[i].y - lower_bound < cutoff) {
            ghost_to_below.push_back(local_parts[i].x);
            ghost_to_below.push_back(local_parts[i].y);
        }
    }

    MPI_Request requests[4];
    int req_count = 0;

    // Send ghost_to_above to rank_above and receive ghost_from_above from rank_above
    if (rank_above != -1) {
        int ghost_to_above_size = ghost_to_above.size();
        MPI_Isend(&ghost_to_above_size, 1, MPI_INT, rank_above, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(ghost_to_above.data(), ghost_to_above_size, MPI_DOUBLE, rank_above, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        int ghost_from_above_size;
        MPI_Irecv(&ghost_from_above_size, 1, MPI_INT, rank_above, 2, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Wait(&requests[req_count - 1], MPI_STATUS_IGNORE);
        ghost_from_above.resize(ghost_from_above_size);
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_size, MPI_DOUBLE, rank_above, 3, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Send ghost_to_below to rank_below and receive ghost_from_below from rank_below
    if (rank_below != -1) {
        int ghost_to_below_size = ghost_to_below.size();
        MPI_Isend(&ghost_to_below_size, 1, MPI_INT, rank_below, 2, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(ghost_to_below.data(), ghost_to_below_size, MPI_DOUBLE, rank_below, 3, MPI_COMM_WORLD, &requests[req_count++]);
        
        int ghost_from_below_size;
        MPI_Irecv(&ghost_from_below_size, 1, MPI_INT, rank_below, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Wait(&requests[req_count - 1], MPI_STATUS_IGNORE);
        ghost_from_below.resize(ghost_from_below_size);
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_size, MPI_DOUBLE, rank_below, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Synchronize before printing
    MPI_Barrier(MPI_COMM_WORLD);

    // Print received vectors after barrier
    std::cout << "[Rank " << rank << "] Received Ghost From Above:\n";
    for (size_t i = 0; i < ghost_from_above.size(); i += 2) {
        std::cout << "  Particle: (x: " << ghost_from_above[i] << ", y: " << ghost_from_above[i + 1] << ")\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] Received Ghost From Below:\n";
    for (size_t i = 0; i < ghost_from_below.size(); i += 2) {
        std::cout << "  Particle: (x: " << ghost_from_below[i] << ", y: " << ghost_from_below[i + 1] << ")\n";
    }
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // Print the x coordinate of the first particle
    std::cout << parts << std::endl;
    if (num_parts > 0) {
        printf("[Rank %d] Gatyer: First particle x = %f\n", rank, parts[0].x);
    }
}