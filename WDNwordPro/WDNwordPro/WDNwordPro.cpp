// WDNwordPro.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <exception>
#include <iostream>
#include <string>
#include "src/STKapp.hpp"

int main()
{
	/*::CoInitialize(NULL);
	IAgSTKXApplicationPtr appStk;
	try
	{
		appStk.CreateInstance(__uuidof(AgSTKXApplication));
	}
	catch (const std::exception e)
	{
		std::cout << e.what() << std::endl;

	}
	::CoUninitialize();*/

	STKapp stkApp;
	stkApp.STKappconfig();
    return 0;
}

