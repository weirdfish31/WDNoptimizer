// WDNwordPro.cpp : �������̨Ӧ�ó������ڵ㡣
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

