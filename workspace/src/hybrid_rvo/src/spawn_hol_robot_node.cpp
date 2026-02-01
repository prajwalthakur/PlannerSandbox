#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <gazebo_msgs/srv/spawn_entity.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

class HolRobotSpawner : public  rclcpp::Node
{
    public:
        HolRobotSpawner(const std::string& configPath, const std::string& sdfPath
        ):
        Node("hol_robot_spawner")
        {
            
            mClient = this->create_client<gazebo_msgs::srv::SpawnEntity>("spawn_entity");
            while(!mClient->wait_for_service(1s))
            {
                RCLCPP_INFO(this->get_logger(),"waiting for /spawn_entity service");
            }
            
            mSdfTemplate = loadSdfFile(sdfPath);
            
            YAML::Node config = YAML::LoadFile(configPath);
            YAML::Node hol_robot_config = config["hol_robots"];
            
            mRadius = hol_robot_config["radius"].as<double>();
            mHeight = hol_robot_config["height"].as<double>();
            mNumRobots = hol_robot_config["num_robots"].as<int>();
            loadRobotConfig(hol_robot_config);
            spawnRobots();
        }
    private:
        std::string loadSdfFile(const std::string& path)
        {
            std::ifstream file(path);
            if(!file.is_open())
            {
                throw std::runtime_error("Failed to open sdf file"+path);
            }
            std::stringstream buffer;
            buffer<<file.rdbuf();
            return buffer.str();
        }
        void loadRobotConfig(YAML::Node& hol)
        {

            for (const auto & p : hol["positions"]) 
            {
                double x = p[0].as<double>();
                double y = p[1].as<double>();
                mPositions.emplace_back(x, y);
            }
                        
            for (const auto & c : hol["colors"])
            {
                if (c.size() != 6) {
                    throw std::runtime_error(
                        "Each color entry must be [x y r g b a]");
                }

                std::ostringstream color;
                color << c[2].as<double>() << " "
                    << c[3].as<double>() << " "
                    << c[4].as<double>() << " "
                    << c[5].as<double>();

                mColors.push_back(color.str());
            }
        }
        void spawnRobots()
        {
            for(int i=0;i<mNumRobots;++i)
            {
                spawnRobot(i);
            }
        }
        static void replaceAll(
            std::string & str,
            const std::string & from,
            const std::string & to)
        {
            size_t pos = 0;
            while ((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
            }
        }
        void spawnRobot(int idx)
        {
            auto request =
                std::make_shared<gazebo_msgs::srv::SpawnEntity::Request>();

            std::string model_name = "robot_" + std::to_string(idx);

            std::string sdf = mSdfTemplate;
            replaceAll(sdf, "${MODEL_NAME}", model_name);
            replaceAll(sdf, "${RADIUS}", std::to_string(mRadius));
            replaceAll(sdf, "${HEIGHT}", std::to_string(mHeight));
            replaceAll(sdf, "${COLOR}", mColors[idx]);

            request->name = model_name;
            request->xml = sdf;
            request->robot_namespace = model_name;
            request->reference_frame = "world";

            geometry_msgs::msg::Pose pose;
            pose.position.x = mPositions[idx].first;
            pose.position.y = mPositions[idx].second;
            pose.position.z = mHeight / 2.0;
            request->initial_pose = pose;

            auto future = mClient->async_send_request(request);
            rclcpp::spin_until_future_complete(
                this->get_node_base_interface(), future);
        }
       
    private:
        std::string mSdfTemplate;
        int mNumRobots{0};
        double mRadius{0.0};
        double mHeight{0.0};
        std::vector<std::pair<double, double>> mPositions;
        std::vector<std::string> mColors;
        rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedPtr mClient;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    if (argc < 3) {
        std::cerr << "Usage:\n"
                << "  spawn_cylinders <config.yaml> <model.sdf>\n";
        return 1;
    }
    std::string config_path = argv[1];
    std::string sdf_path = argv[2];
    auto node = std::make_shared<HolRobotSpawner>(
        config_path, sdf_path);
    rclcpp::shutdown();
    return 0;
}