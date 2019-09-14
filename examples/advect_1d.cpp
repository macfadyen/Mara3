/**
 ==============================================================================
 Copyright 2019, Andrew MacFadyen

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/

/**
 * @brief Solve linear advection equation for scalar field u(x,t) 
 *
 * u_t + div (a * u) = 0
 *
 * where a is a constant advection speed
 */
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>
#include "core_ndarray_ops.hpp"
#include "core_hdf5.hpp"
#include "app_serialize.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"

using namespace nd;

struct state_t
{
	int    iteration;
	double time;
	nd::shared_array<double,1> x_vertices;      
	nd::shared_array<double,1> u; 
};

struct solver_data_t
{
    double advection_speed;
    double cfl;
    double gaussian_width;
    int    rk_order;
};

struct aperture_data_t
{
	double aperture_radius;
	double aperture_speed;
	double aperture_mass;
};

double gaussian (double x)
{
	return std::exp(-0.5 * x * x);
}

state_t initialize_state(const mara::config_t cfg)
{
	auto nx = cfg.get_int( "resolution" );
	auto x_vertices = linspace(cfg.get_double("xmin"), cfg.get_double("xmax"), nx + 1);
	auto x_centers = x_vertices | midpoint_on_axis(0);
	auto u = x_centers | map( [] (auto xi) { return gaussian(xi); } );
	return{0, 0.0, x_vertices.shared(), u.shared()};
}

solver_data_t create_solver_data(const mara::config_t cfg)
{
    return {
        cfg.get_double("advection_speed"),
        cfg.get_double("cfl"),
        cfg.get_double("gaussian_width"),
    };
}

void write_checkpoint(const state_t& state, std::string fname)
{
	auto h5f = h5::File(fname, "w" );

	auto x_centers = state.x_vertices | midpoint_on_axis(0);	

	h5f.write("iteration", state.iteration);
	h5f.write("time",      state.time);
	h5f.write("x_centers", x_centers.shared());
	h5f.write("u",         state.u.shared());

	h5f.close();

	std::cout << "Wrote " << fname << std::endl;
}

auto update_u (const state_t& state, double advection_speed, double dt)
{
	auto u = state.u | extend_periodic_on_axis(0);
	auto F = u * advection_speed;
	auto Fl = F | select_axis(0).from(0).to(1).from_the_end();
	auto Fr = F | select_axis(0).from(1).to(0).from_the_end();	
	auto Fhat = advection_speed > 0.0 ? Fl : Fr;
	auto delta_Fhat =     Fhat | difference_on_axis(0); // defined on faces
	auto dx = state.x_vertices | difference_on_axis(0);

/*	double xa = 0.0;
	double ra = 0.5;
	double aperture_speed = 0.25;
	auto x_centers = state.x_vertices | midpoint_on_axis(0);
	auto aperture_mask = x_centers | map( [] (auto xi) { return (std::abs(xi - 0.0) < 0.5 ? 0.0: 1.0); } );
	auto aperture_normal = aperture_mask | difference_on_axis(0);
	auto u_at_vertex = Fhat / advection_speed;
	auto aperture_mass = aperture_mass - (Fhat - u_at_vertex * aperture_speed) * aperture_normal * dt / dx;
*/
	return state.u - ( delta_Fhat * dt / dx ) | to_shared();

//	auto du_aperture = Fl * dt / dx | to_shared();
}
/*
auto next_state(const state_t& state, const solver_data_t& solver_data)
{
	double dt = 0.025;

	auto s0 = state;

        switch (solver_data.rk_order)
        {
            case 1:
            {
                return    update_u(s0, dt, solver_data.advection_speed);
            }
            case 2:
            {
                auto s1 = update_u(s0, dt, solver_data.advection_speed);
                auto s2 = update_u(s1, dt, solver_data.advection_speed);
                
                return (s0 + s2) * 0.5;
            }
        }
}

solution_t advance(const state_t& state, const solver_data_t& solver_data, double dt)
{
	return{iteration+1, time+dt, x_vertices, update_u(state,dt,advection_speed)}
}
*/

state_t new_state(const state_t& state, const solver_data_t& solver_data)
{
	double dt = 0.01;
	auto    a = solver_data.advection_speed;

	return{state.iteration+1, state.time+dt, state.x_vertices, update_u(state, a, dt)};
}


int main(int argc, const char* argv[])
{
    auto cfg_template = mara::make_config_template()
    .item("resolution",      400)
    .item("tfinal",         10.0)
    .item("xmin",           -5.0)
    .item("xmax",            5.0)
    .item("advection_speed", 0.5)
    .item("gaussian_width",  0.2)
    .item("cfl",             0.5)
    .item("rk_order",          1)    
    .item("ichkpt",            0)        
    .item("outdir", "advect1d_data");

    auto args = mara::argv_to_string_map(argc, argv);
    auto run_config = cfg_template.create().update(args);
    mara::pretty_print(std::cout, "config", run_config);
    auto solver_data = create_solver_data(run_config);
    auto state = initialize_state(run_config);

    while( state.time < run_config.get_double("tfinal") )
    {
    	state = new_state(state, solver_data);
		
    	std::printf("[%06d] time=%3.6lf\n", state.iteration, state.time);	    	

    	if ( state.iteration % 10 == 0 )
    	{
    		auto outdir = run_config.get_string("outdir");
    		auto ichkpt = run_config.get_int   ("ichkpt");    		
    		auto fname  = mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", ichkpt, "h5"));
    	    write_checkpoint(state, fname);
    	    run_config = run_config.set( "ichkpt", ++ichkpt );
    	}
    }


}
