// Implementation of the ClientSocket class

#include <ncs/spec/ClientSocket.h>

namespace ncs {

namespace spec {

ClientSocket::ClientSocket ( std::string host, int port )
{
  if ( ! ncs::spec::Socket::create() )
    {
      throw ncs::spec::SocketException ( "Could not create client socket." );
    }

  if ( ! ncs::spec::Socket::connect ( host, port ) )
    {
      throw ncs::spec::SocketException ( "Could not bind to port." );
    }

}


const ClientSocket& ClientSocket::operator << ( const std::string& s ) const
{
  if ( ! ncs::spec::Socket::send ( s ) )
    {
      throw ncs::spec::SocketException ( "Could not write to socket." );
    }

  return *this;

}


const ClientSocket& ClientSocket::operator >> ( std::string& s ) const
{
  if ( ! ncs::spec::Socket::recv ( s ) )
    {
      throw ncs::spec::SocketException ( "Could not read from socket." );
    }

  return *this;
}

} // namespace spec

} // namespace ncs