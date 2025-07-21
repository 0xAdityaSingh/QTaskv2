using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace HttpClientFactorySample.Services
{
    /// <summary>
    /// Interface for retrieving news stories from a feed URL.
    /// </summary>
    public interface INewsService
    {
        /// <summary>
        /// Retrieves news stories from the specified feed URL.
        /// </summary>
        /// <param name="feedUrl">The URL of the RSS feed.</param>
        /// <returns>A list of news story view models.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="feedUrl"/> is null or empty.</exception>
        /// <exception cref="UriFormatException">Thrown when <paramref name="feedUrl"/> is not a valid URI.</exception>
        /// <exception cref="HttpRequestException">Thrown when an error occurs while making the HTTP request.</exception>
        /// <exception cref="XmlReaderException">Thrown when an error occurs while reading the XML data.</exception>
        Task<List<NewsStoryViewModel>> GetNews(string feedUrl);
    }
}