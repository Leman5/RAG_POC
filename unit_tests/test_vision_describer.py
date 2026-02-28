"""Tests for app/services/vision_describer.py - Vision API service."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import base64


class TestDescribePage:
    """Tests for the describe_page async function."""

    @pytest.mark.asyncio
    async def test_describe_page_success(self):
        """Test successful page description."""
        from app.services.vision_describer import describe_page
        
        # Create mock response
        mock_message = MagicMock()
        mock_message.content = "Screenshot shows dashboard with user settings."
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await describe_page(
                page_image_bytes=b"fake_image_data",
                doc_title="Test Doc",
                page_num=0,
                api_key="test-key",
            )
        
        assert "dashboard" in result.lower()

    @pytest.mark.asyncio
    async def test_describe_page_no_screenshot(self):
        """Test handling when page has no screenshot."""
        from app.services.vision_describer import describe_page
        
        mock_message = MagicMock()
        mock_message.content = "NO_SCREENSHOT"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await describe_page(
                page_image_bytes=b"fake_image_data",
                doc_title="Test Doc",
                page_num=0,
                api_key="test-key",
            )
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_describe_page_handles_error(self):
        """Test error handling in page description."""
        from app.services.vision_describer import describe_page
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client_class.return_value = mock_client
            
            result = await describe_page(
                page_image_bytes=b"fake_image_data",
                doc_title="Test Doc",
                page_num=0,
                api_key="test-key",
            )
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_describe_page_uses_correct_model(self):
        """Test that correct vision model is used."""
        from app.services.vision_describer import describe_page
        
        mock_message = MagicMock()
        mock_message.content = "Description"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            await describe_page(
                page_image_bytes=b"data",
                doc_title="Doc",
                page_num=0,
                api_key="key",
                model="gpt-4-vision-preview",
            )
            
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs['model'] == "gpt-4-vision-preview"


class TestDescribeAllPages:
    """Tests for the describe_all_pages async function."""

    @pytest.mark.asyncio
    async def test_describe_all_pages_empty_input(self):
        """Test handling of empty page_images dict."""
        from app.services.vision_describer import describe_all_pages
        
        result = await describe_all_pages({}, "Doc", "api-key")
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_describe_all_pages_success(self):
        """Test successful description of multiple pages."""
        from app.services.vision_describer import describe_all_pages
        
        mock_message = MagicMock()
        mock_message.content = "Page description"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            page_images = {
                0: b"image1",
                1: b"image2",
            }
            
            result = await describe_all_pages(
                page_images=page_images,
                doc_title="Test Doc",
                api_key="test-key",
            )
        
        assert 0 in result
        assert 1 in result

    @pytest.mark.asyncio
    async def test_describe_all_pages_handles_partial_failures(self):
        """Test handling when some pages fail to describe."""
        from app.services.vision_describer import describe_all_pages
        
        call_count = [0]
        
        async def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First call fails")
            mock_message = MagicMock()
            mock_message.content = "Success"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response
        
        with patch('app.services.vision_describer.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client
            
            page_images = {0: b"img1", 1: b"img2"}
            
            result = await describe_all_pages(
                page_images=page_images,
                doc_title="Doc",
                api_key="key",
            )
        
        # Should have result for page 1 at least
        assert len(result) >= 1


class TestVisionPromptTemplate:
    """Tests for the vision prompt template."""

    def test_prompt_template_has_placeholders(self):
        """Test that prompt template has required placeholders."""
        from app.services.vision_describer import VISION_PROMPT_TEMPLATE
        
        assert "{page_num}" in VISION_PROMPT_TEMPLATE
        assert "{doc_title}" in VISION_PROMPT_TEMPLATE

    def test_prompt_template_content(self):
        """Test prompt template contains expected instructions."""
        from app.services.vision_describer import VISION_PROMPT_TEMPLATE
        
        assert "screenshot" in VISION_PROMPT_TEMPLATE.lower()
        assert "NO_SCREENSHOT" in VISION_PROMPT_TEMPLATE


class TestConcurrencyLimit:
    """Tests for concurrency limiting."""

    def test_max_concurrent_requests_constant(self):
        """Test that MAX_CONCURRENT_REQUESTS is defined."""
        from app.services.vision_describer import MAX_CONCURRENT_REQUESTS
        
        assert isinstance(MAX_CONCURRENT_REQUESTS, int)
        assert MAX_CONCURRENT_REQUESTS > 0
        assert MAX_CONCURRENT_REQUESTS <= 10  # Reasonable limit
