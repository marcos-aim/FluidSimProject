#include "Window.h"

void Window::setupMenuTabs()
{
    ImGui::Begin("Simulation Menu");

    // --- Box Settings ---
    if (ImGui::CollapsingHeader("Box Settings"))
    {
        // Box Size X:
        ImGui::Text("Box Size X:");
        bool changed = ImGui::SliderFloat("##Box Size X Slider", &userInput.boxSizeX, 0.1f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changed |= ImGui::InputFloat("##Box Size X Input", &userInput.boxSizeX, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Box Size Y:
        ImGui::Text("Box Size Y:");
        changed |= ImGui::SliderFloat("##Box Size Y Slider", &userInput.boxSizeY, 0.1f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changed |= ImGui::InputFloat("##Box Size Y Input", &userInput.boxSizeY, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Box Size Z:
        ImGui::Text("Box Size Z:");
        changed |= ImGui::SliderFloat("##Box Size Z Slider", &userInput.boxSizeZ, 0.1f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changed |= ImGui::InputFloat("##Box Size Z Input", &userInput.boxSizeZ, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        if (changed)
        {
            rendererWindow->prepareBoxBuffers(userInput.boxSizeX, userInput.boxSizeY, userInput.boxSizeZ);
            simulation->updateParameters(userInput);
        }
    }

    // --- Particle Settings ---
    if (ImGui::CollapsingHeader("Particle Settings"))
    {
        bool changedParticle = false;
        // Particle Radius:
        ImGui::Text("Particle Radius:");
        changedParticle |= ImGui::SliderFloat("##Particle Radius Slider", &userInput.particleR, 0.01f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedParticle |= ImGui::InputFloat("##Particle Radius Input", &userInput.particleR, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Sphere Slices:
        ImGui::Text("Sphere Slices:");
        changedParticle |= ImGui::SliderInt("##Sphere Slices Slider", &userInput.sphereSlices, 4, 100);
        ImGui::SameLine();
        changedParticle |= ImGui::InputInt("##Sphere Slices Input", &userInput.sphereSlices);

        // Sphere Stacks:
        ImGui::Text("Sphere Stacks:");
        changedParticle |= ImGui::SliderInt("##Sphere Stacks Slider", &userInput.sphereStacks, 4, 100);
        ImGui::SameLine();
        changedParticle |= ImGui::InputInt("##Sphere Stacks Input", &userInput.sphereStacks);

        if (changedParticle)
        {
            rendererWindow->prepareSphereBuffers(userInput.particleR,
                                                   userInput.sphereSlices,
                                                   userInput.sphereStacks,
                                                   rendererWindow->sphereTransforms);
        }
    }

    // --- SPH Settings ---
    if (ImGui::CollapsingHeader("SPH Settings"))
    {
        bool changedSPH = false;
        // Particle Count:
        ImGui::Text("Particle Count:");
        changedSPH |= ImGui::SliderInt("##Particle Count Slider", &userInput.particleCount, 0, 50000);
        ImGui::SameLine();
        changedSPH |= ImGui::InputInt("##Particle Count Input", &userInput.particleCount);

        // Resting Density:
        ImGui::Text("Resting Density:");
        changedSPH |= ImGui::SliderFloat("##Resting Density Slider", &userInput.restingDensity, 0.0f, 5000.0f, "%.1f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Resting Density Input", &userInput.restingDensity, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Viscosity Multiplier:
        ImGui::Text("Viscosity Multiplier:");
        changedSPH |= ImGui::SliderFloat("##Viscosity Multiplier Slider", &userInput.viscosityMultiplier, 0.0f, 50.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Viscosity Multiplier Input", &userInput.viscosityMultiplier, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Mass:
        ImGui::Text("Mass:");
        changedSPH |= ImGui::SliderFloat("##Mass Slider", &userInput.mass, 0.0f, 10.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Mass Input", &userInput.mass, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Gas Constant:
        ImGui::Text("Gas Constant:");
        changedSPH |= ImGui::SliderFloat("##Gas Constant Slider", &userInput.gasConstant, 0.0f, 50.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Gas Constant Input", &userInput.gasConstant, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Smoothing Radius (h):
        ImGui::Text("Smoothing Radius (h):");
        changedSPH |= ImGui::SliderFloat("##Smoothing Radius Slider", &userInput.h, 0.0f, 5.0f, "%.3f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Smoothing Radius Input", &userInput.h, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Gravity (g):
        ImGui::Text("Gravity (g):");
        changedSPH |= ImGui::SliderFloat("##Gravity Slider", &userInput.g, -50.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Gravity Input", &userInput.g, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        // Surface Tension:
        ImGui::Text("Surface Tension:");
        changedSPH |= ImGui::SliderFloat("##Surface Tension Slider", &userInput.tension, 0.0f, 5.0f, "%.2f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedSPH |= ImGui::InputFloat("##Surface Tension Input", &userInput.tension, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        if (ImGui::Button("Reset to Defaults"))
        {
            userInput.restingDensity      = 1000.0f;
            userInput.viscosityMultiplier = 1.0f;
            userInput.mass                = 0.2f;
            userInput.gasConstant         = 1.0f;
            userInput.h                   = 0.15f;
            userInput.g                   = -9.8f;
            userInput.tension             = 0.2f;

            simulation->updateParameters(userInput);
        }

        if (changedSPH) {
            simulation->updateParameters(userInput);
        }
    }

    // --- Simulation Controls ---
    if (ImGui::CollapsingHeader("Simulation Controls"))
    {
        if (ImGui::Checkbox("Run Simulation", &userInput.runSimulation))
        {
            if (userInput.runSimulation)
                std::cout << "Simulation started." << std::endl;
            else
                std::cout << "Simulation paused." << std::endl;
        }

        // Add slider for time step (dt)
        ImGui::Text("Time Step (dt):");
        bool changedDt = ImGui::SliderFloat("##Delta Time Slider", &userInput.dt, 0.001f, 1.0f, "%.3f");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        changedDt |= ImGui::InputFloat("##Delta Time Input", &userInput.dt, 0.0f, 0.0f, "%.3f");
        ImGui::PopStyleColor();

        if (changedDt)
        {
            // Optionally update simulation parameters immediately if needed
            simulation->updateParameters(userInput);
        }
    }


    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);

    ImGui::End();
}
