#include <iostream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
using namespace std;

const int BUF_MAX_SIZE = 2048;

void readALL(int connfd, string &msg)
{
    // echo msg
    size_t t;
    char buf[BUF_MAX_SIZE];
    while (1)
    {
        int n = read(connfd, buf, BUF_MAX_SIZE);
        string temp(buf, buf + n);
        if (n < BUF_MAX_SIZE)
        {
            msg = msg + temp;
            break;
        }
        else
            msg = msg + temp;
    }
}

void str_send(int sockfd)
{

    string msg;
    getline(cin, msg);
    write(sockfd, msg.c_str(), msg.size());
    msg="";
    readALL(sockfd, msg);
    cout << msg << endl;
}

int main(int argv, char *args[])
{

    int sockfd;
    sockaddr_in serveraddr;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    memset(&serveraddr, 0, sizeof(serveraddr));

    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = inet_addr(args[1]);
    serveraddr.sin_port = htons(atoi(args[2]));

    if (connect(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == -1)
    {
        cerr << "connect err" << endl;
    }

    str_send(sockfd);
    close(sockfd);
}