#include <ncs/spec/ClientSocket.h>

namespace ncs {

namespace spec {

ClientSocket::ClientSocket ()
{
  if (!ncs::spec::Socket::create())
    {
      std::cout << "Could not create client socket.\n";
    }
}

ClientSocket::ClientSocket (std::string host, int port)
{
  if (!ncs::spec::Socket::create())
    {
      throw ncs::spec::SocketException ("Could not create client socket.");
    }

  if (!ncs::spec::Socket::connect(host, port))
    {
      throw ncs::spec::SocketException("Could not bind to port.");
    }
}

bool ClientSocket::bindWithoutThrow(std::string host, int port)
{
  if ( !ncs::spec::Socket::connect(host, port) )
  {
    std::cout << "Could not bind to port.\n";
    return false;
  }
  return true;
}

const ClientSocket& ClientSocket::operator << (const std::string& s) const
{
  if (!ncs::spec::Socket::send(s))
    {
      throw ncs::spec::SocketException("Could not write to socket.");
    }
  return *this;
}

const ClientSocket& ClientSocket::operator >> (std::string& s) const
{
  if (!ncs::spec::Socket::recv(s))
    {
      throw ncs::spec::SocketException("Could not read from socket.");
    }
  return *this;
}

void ClientSocket::close()
{
  if (!ncs::spec::Socket::close())
    {
      throw ncs::spec::SocketException("Could not close client socket");
    }
}

} // namespace spec

} // namespace ncs