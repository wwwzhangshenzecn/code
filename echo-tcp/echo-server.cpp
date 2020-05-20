#include <iostream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include<sys/wait.h>
#include <string.h>
#include <stdlib.h>
#include<signal.h>
using namespace std;


const int SERVER_PORT = 12345;

const int BUF_MAX_SIZE = 2048;

void sig_child(int signo){
    pid_t pid;
    int stat;

    while((pid = waitpid(-1, &stat, WNOHANG)) > 0){
        cout<<"child "<<pid<<" teminated"<<endl;
    }
    return;
}



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

void str_echo(int connfd)
{
    string msg;
    readALL(connfd, msg);
    write(connfd, msg.c_str(), msg.size());
}

int main(int argc, char *args[])
{

    int listenfd, connfd;
    pid_t pid;

    socklen_t childlen;
    sockaddr_in clientaddr, serveraddr;

    listenfd = socket(AF_INET, SOCK_STREAM, 0);

    if (listenfd == -1)
    {
        cerr << "socket() err" << endl;
        return 0;
    }

    memset(&serveraddr, 0, sizeof(serveraddr));

    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons(atoi(args[1]));

    if (bind(listenfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == -1)
    {
        cerr << "bind err" << endl;
        return 0;
    }
    cout << "bind ..." << endl;
    if (listen(listenfd, 5) == -1)
    {
        cerr << "listen err" << endl;
        return 0;
    }
    cout << "listen ..." << endl;
    signal(SIGCHLD, sig_child);
    while (1)
    {
        sleep(3);
        cout << "accpet ..." << endl;
        childlen = sizeof(clientaddr);
        connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &childlen);
        if ((pid = fork()) == 0)
        {
            //child process
            close(listenfd);
            str_echo(connfd);
            return 0;
        }
        close(connfd);
    }
}