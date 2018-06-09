#ifndef STK_APP_HPP
#define STK_APP_HPP

#include "..\stdafx.h"
#include <string>
#include <map>
#include <list>
#include <vector>
#include <iostream>
#include <memory>
#include <fstream>
//#include <..\boostlib\algorithm\string.hpp>
#include <sstream>    //使用stringstream需要引入这个头文件  
using namespace std;
//模板函数：将string类型变量转换为常用的数值类型（此方法具有普遍适用性）  
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

/*
 *this is a class to create a STK application
 *and then can config out some config file to simulation for communication
 **/
typedef std::map<std::string, int> NameId;
typedef std::map<std::string, int> MobileName;
typedef _bstr_t stkString;
//bool Stkflag = false;
typedef struct GroundNode
{
	std::string nodeName;
	double		Lat;
	double		Lon;
};
typedef struct MobileNode
{
	std::string nodeName;
};

typedef std::list<GroundNode> GNodelist;
typedef std::list<MobileNode> MNodelist;
typedef struct SatWalker
{
	std::string rootSatName;
	double helight;
	double RAAN;
	int numPlans;
	int numSatOfPlans;
	double incline;
	double prasePara;
	std::string starttime;
	std::string endtime;
	double RAANrange;
	std::string ScenName;
};

stkString to_stkString(std::string str)
{
	stkString toStr(str.c_str());
	return toStr;
}

class STKapp
{
public:
	STKapp()
	{
		::CoInitialize(NULL);
		::CoInitialize(NULL);
		appStk.CreateInstance(__uuidof(AgSTKXApplication));
	}
	~STKapp()
	{
		appStk.Release();
		::CoUninitialize();
	}
	void STKappconfig()
	{
		assert(0 == readGnodeFile("../configfile/GroundNode.Position"));
		assert(0 == readMnodeFile("../configfile/MobileNode.name"));
		assert(0 == readWalkerFile("../configfile/Scenary.sat", this->COMwalkerPara));
		assert(0 == readWalkerFile("../configfile/ScenaryRS.sat", this->RSwalkerPara));
		
		assert(0 == appStkInit());
		assert(0 == preScenBuild("Scenario"));
		assert(0 == GroundNodeBuild());
		assert(0 == MobileNodeBuild());
		
		WalkerBuild(this->COMwalkerPara);
		WalkerBuild(this->RSwalkerPara, false, true);
		AccessOut();
		if (false)
		{
			Qimapout("../OutConfigfile/WDNscenario.qimap", this->COMwalkerPara);
			std::cout << "..... out the qimap file ...\n";
			assert(0 == Vdffileout("WDNscenario"));
			std::cout << "..... out the vdf file ...\n  end....... \n";
		}
		else 
		{
			NodePositionOut();
		}
		std::cout << "..... end the build file ...\n  end....... \n";
	}
//protected:
private:
	IAgSTKXApplicationPtr appStk;
	NameId NodeNameId;
	GNodelist GNode;
	MNodelist MNode;

	std::list<std::string> RsSat;
	int readGnodeFile(std::string filename)
	{
		std::cout << "..... start read the Ground nodes ...";
		std::ifstream Gnodefile;
		try
		{
			Gnodefile.open(filename.c_str());
			assert(Gnodefile.is_open());
			while (!Gnodefile.eof())
			{
				char buffer[256];
				Gnodefile.getline(buffer, 256);
				//temstr(buffer);
				if ( buffer[0] == '#') 
				{
					continue;
				}
				std::string temstr(buffer);
				std::string sbstr = " ";
				std::vector<std::string> stemstr;
				while (temstr.length())
				{
					std::string tempstr;
					tempstr = temstr.substr(0, temstr.find(sbstr));
					if (tempstr.length() == 0) 
					{
						temstr.replace(0, 1, "");
					}
					else
					{
						temstr.replace(0, tempstr.length(), "");
					}
					if (tempstr.length() >0 && tempstr != " ")
					{
						stemstr.push_back(tempstr);
					}
				}		
				if (stemstr.size()> 0 && stemstr.size() != 3) 
				{
					std::cout << "the Gnode file is wrong\n";
				}
				if (stemstr.size() > 0) {
					assert(stemstr.size() == 3);

					GroundNode tempNode;
					tempNode.nodeName = stemstr[0];
					tempNode.Lat = stringToNum<double>(stemstr[1]);
					tempNode.Lon = stringToNum<double>(stemstr[2]);
					GNode.push_back(tempNode);
					//std::cout << tempNode.nodeName << "  ";
				}
			}			
			Gnodefile.close();
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "read the Ground Node file is failed\n";
			return -1;
		}
	}
	int readMnodeFile(std::string filename)
	{
		std::cout << "\n..... start read the Mobile nodes ...";
		std::ifstream Mnodefile;
		try
		{
			Mnodefile.open(filename.c_str());
			assert(Mnodefile.is_open());
			while (!Mnodefile.eof())
			{
				char buffer[256];
				Mnodefile.getline(buffer, 256);
				//temstr(buffer);
				if (buffer[0] == '#') 
				{
					continue;
				}
				std::string temstr(buffer);
				std::string sbstr = " ";
				std::vector<std::string> stemstr;
				while (temstr.length())
				{
					std::string tempstr;
					tempstr = temstr.substr(0, temstr.find(sbstr));
					if (tempstr.length() == 0) 
					{
						temstr.replace(0, 1, "");
					}
					else
					{
						temstr.replace(0, tempstr.length(), "");
					}
					if (tempstr.length() >0 && tempstr != " ") 
					{
						stemstr.push_back(tempstr);
					}
				}
				if (stemstr.size()> 0 && stemstr.size() != 1) 
				{
					std::cout << "the MobileNode file is wrong\n";
				}
				if (stemstr.size() > 0) {

					MobileNode tempNode;
					tempNode.nodeName = stemstr[0];
					MNode.push_back(tempNode);
					//std::cout << tempNode.nodeName << "  " ;
				}
			}
			Mnodefile.close();
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "read the Mobile Node file is failed\n";
			return -1;
		}
	}
	int readWalkerFile(std::string filename, SatWalker & walkerPara)
	{
		std::string rootname("rootname");
		std::string helight("helight");
		std::string numPlans("numPlans");
		std::string numSatOfPlans("numSatOfPlans");
		std::string incline("incline");
		std::string prasePara("prasePara");
		std::string starttime("starttime");
		std::string endtime("endtime");
		std::string RAANrange("RAANrange");
		std::string ScenName("ScenName");
		std::string RAANName("RAAN");
		std::ifstream configWalker;
		try
		{
			configWalker.open(filename);
			assert(configWalker.is_open());
			while (!configWalker.eof())
			{
				char buffer[256];
				configWalker.getline(buffer, 256);
				//temstr(buffer);
				if (buffer[0] == '#') 
				{
					continue;
				}
				std::string tempstr(buffer);
				if (tempstr.find(rootname) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(rootname), rootname.length() + 1, "");
					walkerPara.rootSatName = tempstr;
				}
				if (tempstr.find(helight) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(helight), helight.length() + 1, "");
					walkerPara.helight = stringToNum<double>(tempstr);
				}
				if (tempstr.find(numPlans) != std::string::npos)
				{
					tempstr.replace(tempstr.find(numPlans), numPlans.length() + 1, "");
					walkerPara.numPlans = stringToNum<int>(tempstr);
				}
				if (tempstr.find(numSatOfPlans) != std::string::npos)
				{
					tempstr.replace(tempstr.find(numSatOfPlans), numSatOfPlans.length() + 1, "");
					walkerPara.numSatOfPlans = stringToNum<int>(tempstr);
				}
				if (tempstr.find(incline) != std::string::npos)
				{
					tempstr.replace(tempstr.find(incline), incline.length() + 1, "");
					walkerPara.incline = stringToNum<double>(tempstr);
				}
				if (tempstr.find(prasePara) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(prasePara), prasePara.length() + 1, "");
					walkerPara.prasePara = stringToNum<double>(tempstr);
				}
				if (tempstr.find(starttime) != std::string::npos)
				{
					tempstr.replace(tempstr.find(starttime), starttime.length() + 1, "");
					walkerPara.starttime = tempstr;
				}
				if (tempstr.find(endtime) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(endtime), endtime.length() + 1, "");
					walkerPara.endtime = tempstr;
				}
				if (tempstr.find(RAANrange) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(RAANrange), RAANrange.length() + 1, "");
					walkerPara.RAANrange = stringToNum<double>(tempstr);
				}

				if (tempstr.find(ScenName) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(ScenName), ScenName.length() + 1, "");
					walkerPara.ScenName = (tempstr);
				}
				if (tempstr.find(RAANName) != std::string::npos) 
				{
					tempstr.replace(tempstr.find(RAANName), RAANName.length() + 1, "");
					walkerPara.RAAN = stringToNum<double>(tempstr);
				}
			}
			configWalker.close();
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "read the Walker file failed\n";
			return -1;
		}
	}

	SatWalker COMwalkerPara;
	bool rswalker;
	SatWalker RSwalkerPara;
	
	int appStkInit()
	{
		try
		{
			appStk->put_EnableConnect(VARIANT_TRUE);
			appStk->put_ConnectPort(5525);
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "application init is wrong\n";
			return -1;
		}
	}
	int preScenBuild(std::string ScenName)
	{
		std::cout << "\n..... start creat the stk config scenario ...";
		stkString Sname = to_stkString(ScenName);
		stkString cmdStr = "New / Scenario ";
		try
		{
			appStk->ExecuteCommand(cmdStr + Sname );
			//std::cout << "the Scenario " << ScenName << " is built";
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "the Scenario build is failed\n";
			return -1;
		}
	}
	int GroundNodeBuild()
	{
		std::cout << "\n..... creat the stk ground node ...\n";
		stkString cmdStr_new = "New / */Facility  ";
		stkString cmdStr_setPosition = "SetPosition */Facility/";
		try
		{
			for (auto itr : GNode) 
			{
				stkString tempstkstr = to_stkString(itr.nodeName);
				appStk->ExecuteCommand(cmdStr_new + tempstkstr);
				std::string tempstr = std::to_string(itr.Lat) + " " + std::to_string(itr.Lon);
				stkString setstkString = cmdStr_setPosition + tempstkstr + " Geodetic  " + to_stkString(tempstr) + " 0.0";
				appStk->ExecuteCommand(setstkString);
				bool Stkflag = false;
				if (Stkflag)
				{
					std::string anpre = "an";
					for (int i = 0; i < 20; i++) 
					{
						stkString anName = to_stkString(anpre) + to_stkString(std::to_string(i));
						stkString newAn = "New / */Facility/" + tempstkstr + "/Antenna " + anName;
						appStk->ExecuteCommand(newAn);
					}
				}
				NodeNameId.insert(std::make_pair(itr.nodeName, 0));
			}
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "the build Ground Node is failed\n";
			return -1;
		}
	}
	int MobileNodeBuild()
	{
		stkString loadcmd = "Load / */Aircraft \"G:\\WDNoptimizer\\WDNwordPro\\configfile\\mobilenodes\\";
		stkString suffix = ".ac\"";
		try
		{
			for (auto itr : MNode) 
			{
				stkString tempstkstr = to_stkString(itr.nodeName);
				appStk->ExecuteCommand(loadcmd + tempstkstr + suffix);
 				NodeNameId.insert(std::make_pair(itr.nodeName, 3));
				//std::cout << "the Mobile Node " << itr.nodeName << " is loaded \n";
			}
			return 0;

		}
		catch (const std::exception& e)
		{
			std::cout << "the build Mobile Node is failed\n";
			return -1;
		}
	}
	int WalkerBuild(SatWalker walkerPara, bool Stkflag =false, bool RS = false)
	{
		try
		{
			stkString starttimeStr = " \"" + to_stkString(walkerPara.starttime) + "\" ";
			stkString endtimeStr = " \"" + to_stkString(walkerPara.endtime) + "\" ";
			stkString settimeStr = "SetAnalysisTimePeriod *";
			stkString epochStr = "SetEpoch * ";
			std::cout << settimeStr + starttimeStr + endtimeStr << "\n";
			stkString newSatStr = "New / */Satellite/ " + to_stkString(walkerPara.rootSatName);
			appStk->ExecuteCommand(newSatStr);
			stkString setSatStateStr = "SetState */Satellite/" + to_stkString(walkerPara.rootSatName) + " Classical J2Perturbation "
				+ starttimeStr + endtimeStr + " 60 J2000 " + starttimeStr + to_stkString(std::to_string(6370000.0 + 1000.0 * walkerPara.helight))
				+ " 0.0 " + to_stkString(std::to_string(walkerPara.incline)) 
				+ " 0.0 " + to_stkString(std::to_string(walkerPara.RAAN)) +" 360";
			std::cout << setSatStateStr << "\n";
			appStk->ExecuteCommand(setSatStateStr);
			if (Stkflag)
			{
				std::string anpre = "an";
				for (int i = 0; i < 20; i++) 
				{
					stkString anName = to_stkString(anpre) + to_stkString(std::to_string(i));
					stkString newAn = "New / */Satellite/" + to_stkString(walkerPara.rootSatName) + "/Antenna " + anName;
					appStk->ExecuteCommand(newAn);
				}
				if (RS == false)
				{
					stkString SatCons = "SetConstraint */Satellite/" + to_stkString(walkerPara.rootSatName) + " ElevationAngle Max -35";
					appStk->ExecuteCommand(SatCons);
				}
			}
			stkString walkerStr = "Walker */Satellite/" + to_stkString(walkerPara.rootSatName)
				+ " " + to_stkString(std::to_string(walkerPara.numPlans))
				+ " " + to_stkString(std::to_string(walkerPara.numSatOfPlans))
				+ " " + to_stkString(std::to_string(walkerPara.prasePara))
				+ " " + to_stkString(std::to_string(walkerPara.RAANrange))
				+ " Yes";
			appStk->ExecuteCommand(walkerStr);
			appStk->ExecuteCommand(settimeStr + starttimeStr + endtimeStr);
			appStk->ExecuteCommand(epochStr + starttimeStr);
			
			stkString unloadSat = "Unload  / */Satellite/" + to_stkString(walkerPara.rootSatName);
			appStk->ExecuteCommand(unloadSat);
			if (RS)
			{
				std::ifstream Rsource;
				Rsource.open("../configfile/Source.RSsat");
				for (std::string str; std::getline(Rsource, str);) 
				{
					NodeNameId.insert(std::make_pair(str, 2));
					RsSat.push_back(str);
				}
				std::cout << "..... Remote Sensor walker is built ...\n";
			}
			else 
			{
				for (int i = 1; i <= walkerPara.numPlans; i++)
				{
					for (int j = 1; j <= walkerPara.numSatOfPlans; j++)
					{
						std::string Planstr = std::to_string(i);
						if (walkerPara.numPlans > 9 && i <= 9) 
						{
							Planstr = "0" + Planstr;
						}
						std::string satstr = std::to_string(j);
						if (walkerPara.numSatOfPlans > 9 && j <= 9) 
						{
							satstr = "0" + satstr;
						}
						std::string satName = walkerPara.rootSatName + Planstr + satstr;
						NodeNameId.insert(std::make_pair(satName, 1));
					}
				}
				std::cout << "..... the ComSat walker is built ...\n";
			}
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "the Walker building is wrong\n";
			return -1;
		}
	}
	int AccessOut()
	{
		std::cout << "\n"<<"accessing out\n";		
		try
		{
			appStk->ExecuteCommand("SetUnits / EpochSec");			
			
			for (auto itr_Gnode : GNode)
			{
				stkString reportPre = "Report */Facility/";
				reportPre = reportPre + to_stkString(itr_Gnode.nodeName);
				for (auto itr_Sat : NodeNameId) 
				{
					if (itr_Sat.second == 1) 
					{
						stkString satRrp = " */Satellite/";
						satRrp = satRrp + to_stkString(itr_Sat.first);
						stkString cmdRpStr = reportPre + " SaveAs \"Access\" \"G:\\WDNoptimizer\\WDNwordPro\\outAccess\\"
							+ to_stkString( itr_Gnode.nodeName + "-" + itr_Sat.first + ".csv\"") + satRrp;
						//std::cout << cmdRpStr << "\n";
						appStk->ExecuteCommand(cmdRpStr);
					}
				}
			}
			std::cout << "the Ground Nodes\' csv is completed \n";
			for (auto itr_RS : RsSat)
			{
				stkString reportPre = "Report */Satellite/";
				reportPre = reportPre + to_stkString(itr_RS);
				for (auto itr_Sat : NodeNameId) 
				{
					if (itr_Sat.second == 1) 
					{
						stkString satRrp = " */Satellite/";
						satRrp = satRrp + to_stkString(itr_Sat.first);
						stkString cmdRpStr = reportPre + " SaveAs \"Access\" \"G:\\WDNoptimizer\\WDNwordPro\\outAccess\\"
							+ to_stkString(itr_RS + "-" + itr_Sat.first + ".csv\"") + satRrp;
						//std::cout << cmdRpStr << "\n";
						appStk->ExecuteCommand(cmdRpStr);
					}
				}
			}
			std::cout << "the RS satellites\' csv is completed \n";
			for (auto itr_Mnode : MNode) 
			{
				stkString reportPre = "Report */Aircraft/";
				reportPre = reportPre + to_stkString(itr_Mnode.nodeName);
				for (auto itr_Sat : NodeNameId) 
				{
					if (itr_Sat.second == 1) 
					{
						stkString satRrp = " */Satellite/";
						satRrp = satRrp + to_stkString(itr_Sat.first);
						stkString cmdRpStr = reportPre + " SaveAs \"Access\" \"G:\\WDNoptimizer\\WDNwordPro\\outAccess\\"
							+ to_stkString(itr_Mnode.nodeName+ "-" + itr_Sat.first + ".csv\"") + satRrp;
						//std::cout << cmdRpStr << "\n";
						appStk->ExecuteCommand(cmdRpStr);
					}
				}
			}
			std::cout << "the Mobile Nodes\' csv is completed \n";
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "the AccessOut building is wrong\n";
			return -1;
		}
	}
	int NodePositionOut()
	{
		std::ofstream nodeconfig;
		std::ofstream nodes;
		std::ifstream nodetxt;
		nodeconfig.open("G:/WDNoptimizer/WDNwordPro/OutConfigfile/Scenario.nodeconfig");
		nodeconfig.clear();
		nodes.open("G:/WDNoptimizer/WDNwordPro/OutConfigfile/Scenario.nodes");
		nodes.clear();
		//stkString reportCmd = "ReportCreate */Satellite/Satname Type Save Style Stylename SaveFilePath Timeperiod TimeStep ";
		std::string preFold = "G:\\WDNoptimizer\\WDNwordPro\\nodetxt\\";
		int i = 0;	
		for (auto itr : NodeNameId){
			i++;
			nodeconfig << "[" << i << "] HOSTNAME " << itr.first << "\n";
			std::string fileString = preFold + itr.first;
			if (itr.second == 0){
				
				stkString reportCmd = "ReportCreate */Facility/" + to_stkString(itr.first)
					+ " Type Save "
					+ " Style \"G:\\WDNoptimizer\\WDNwordPro\\WDNwordPro\\Style\\FixLLA.rst\" "
					+ " File \"" + to_stkString(fileString) + ".txt\" "
					//+ " TimePeriod " 
					//+ " \""+to_stkString(walkerPara.starttime)+ "\" " 
					//+ " \""+to_stkString(walkerPara.endtime) + "\" " 
					+ " TimeStep 30";
				//std::cout << reportCmd << std::endl;
				appStk->ExecuteCommand(reportCmd);
				//write the .nodes file
				nodetxt.open(fileString+".txt");
				int line = 1;
				for (std::string str ; std::getline(nodetxt, str); line++)
				{
					if (line >= 7 && str.length() > 0)
					{
						nodes << i << " 0.0S (" << str << ")\n";
					}
				}
				nodetxt.close();
			}
			if (itr.second == 1 || itr.second == 2)
			{

				stkString reportCmd = "ReportCreate */Satellite/" + to_stkString(itr.first)
					+ " Type Save"
					+ " Style \"G:\\WDNoptimizer\\WDNwordPro\\WDNwordPro\\Style\\SatLLA.rst\" "
					+ " File \"" + to_stkString(fileString) + ".txt\" "
					//+ " TimePeriod " 
					//+ " \"" + to_stkString(walkerPara.starttime)+ "\" " 
					//+ " \"" + to_stkString(walkerPara.endtime)+ "\" " 
					+ " TimeStep 30";
				//std::cout << reportCmd << std::endl;
				appStk->ExecuteCommand(reportCmd);
				//write the nodes file
				nodetxt.open(fileString + ".txt");
				int line = 1;
				for (std::string str; std::getline(nodetxt, str); line++)
				{
					if (line >= 7 && str.length() > 20)
					{
						str[12] = 'S';
						str[15] = '(';
						nodes << i  << str << ")\n";
					}
				}
				nodetxt.close();
			}
			if (itr.second == 3)
			{
				stkString reportCmd = "ReportCreate */Aircraft/" + to_stkString(itr.first)
					+ " Type Save"
					+ " Style \"G:\\WDNoptimizer\\WDNwordPro\\WDNwordPro\\Style\\AircraftLLA.rst\" "
					+ " File \"" + to_stkString(fileString) + ".txt\" "
					//+ " TimePeriod " 
					//+ " \"" + to_stkString(walkerPara.starttime)+ "\" " 
					//+ " \"" + to_stkString(walkerPara.endtime)+ "\" " 
					+" TimeStep 30";
				//std::cout << reportCmd << std::endl;
				appStk->ExecuteCommand(reportCmd);
				//write the nodes file
				nodetxt.open(fileString + ".txt");
				int line = 1;
				for (std::string str; std::getline(nodetxt, str); line++)
				{
					if (line >= 7 && str.length() > 20)
					{
						str[12] = 'S';
						str[15] = '(';
						nodes << i << str << ")\n";
					}
				}
				nodetxt.close();
			}
		}

		nodes.close();
		//NODE-POSITION-FILE
		nodeconfig << "#\n";
		nodeconfig << "[" << 1 << " thru " << NodeNameId.size() << "] NODE-PLACEMENT FILE\n";
		nodeconfig << "[" << 1 << " thru " << NodeNameId.size() << "] MOBILITY FILE\n";
		nodeconfig << "NODE-POSITION-FILE Scenario.nodes\n";
		nodeconfig.close();
		return 0;

	}

	int Qimapout(std::string filename, SatWalker walkerPara)
	{
		std::ofstream qimapFile;
		std::ofstream nodeconfig;

		try
		{
			nodeconfig.open("G:/WDNoptimizer/WDNwordPro/OutConfigfile/Scenario.nodeconfig");
			assert(nodeconfig.is_open());
			nodeconfig.clear();
			qimapFile.open(filename);
			assert(qimapFile.is_open());
			qimapFile.clear();
			qimapFile << "Begin Entities\n";
			int i = 0;
			for (auto itr : NodeNameId)
			{
				i++;				
				nodeconfig << "[" << i << "] HOSTNAME " << itr.first << "\n";
				if (itr.second == 1) 
				{
					qimapFile
						<< "    " << "Begin Entity\n"
						<< "        " << "EntityId    " << i << "\n"
						<< "        " << "STKPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Satellite/" << itr.first << "\n"
						<< "        " << "IsSatellite False\n";
					for (int j = 0; j < 20; j++) 
					{
						qimapFile
							<< "        " << "Begin Interface\n"
							<< "             " << "InterfaceId    " << j << "\n"
							<< "             " << "StkTransmitterPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Satellite/" << itr.first << "/Antenna/an" << j << "\n"
							<< "             " << "StkReceiverPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Satellite/" << itr.first << "/Antenna/an" << j << "\n"
							<< "             " << "RainOutagePercent       0.1\n"
							<< "        " << "End Interface\n";
					}
					qimapFile
						<< "    " << "End Entity\n";
				}
				if (itr.second == 0)
				{
					qimapFile
						<< "    " << "Begin Entity\n"
						<< "        " << "EntityId    " << i << "\n"
						<< "        " << "STKPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Facility/" << itr.first << "\n"
						<< "        " << "IsSatellite False\n";
					for (int j = 0; j < 20; j++)
					{
						qimapFile
							<< "        " << "Begin Interface\n"
							<< "             " << "InterfaceId    " << j << "\n"
							<< "             " << "StkTransmitterPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Facility/" << itr.first << "/Antenna/an" << j << "\n"
							<< "             " << "StkReceiverPath    /Application/STK/Scenario/" << walkerPara.ScenName << "/Facility/" << itr.first << "/Antenna/an" << j << "\n"
							<< "             " << "RainOutagePercent       0.1\n"
							<< "        " << "End Interface\n";
					}
					qimapFile
						<< "    " << "End Entity\n";
				}
			}			
			qimapFile << "End Entities\n";
			qimapFile.close();
			nodeconfig.close();
			std::cout << "the qimap file is built\n";
			return 0;
		}
		catch (const std::exception& e)
		{
			std::cout << "the Qimapout building is wrong\n";
			return -1;
		}
	}
	int Vdffileout(std::string filname)
	{
		stkString cmdPre = "Author * CreateViewerDataFile \"G:\\WDNoptimizer\\WDNwordPro\\OutConfigfile\" ";
		stkString vdfName = to_stkString(filname);
		stkString suffixname = ".vdf";
		try
		{
			appStk->ExecuteCommand(cmdPre + vdfName + suffixname);
			std::cout << "the vdf file is built\n";
			return 0;
		}
		catch (const std::exception& e)
		{
			return -1;
		}		
	}
};



#endif